"""
TensorFlow policy class used for PPO.
"""

import gym
import logging
from typing import Dict, List, Optional, Type, Union

import ray
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import (
    compute_gae_for_sample_batch,
    Postprocessing,
)
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import (
    EntropyCoeffSchedule,
    LearningRateSchedule,
    KLCoeffMixin,
    ValueNetworkMixin,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, get_variable
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.utils.typing import (
    AgentID,
    LocalOptimizer,
    ModelGradients,
    TensorType,
    TrainerConfigDict,
)

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)


def validate_config(config: TrainerConfigDict) -> None:
    """Executed before Policy is "initialized" (at beginning of constructor).
    Args:
        config (TrainerConfigDict): The Policy's config.
    """
    # If vf_share_layers is True, inform about the need to tune vf_loss_coeff.
    if config.get("model", {}).get("vf_share_layers") is True:
        logger.info(
            "`vf_share_layers=True` in your model. "
            "Therefore, remember to tune the value of `vf_loss_coeff`!"
        )


# We need this builder function because we want to share the same
# custom logics between TF1 static and TF2 eager policies.
def get_ppo_tf_policy(base: type) -> type:
    """Construct a PPOTFPolicy inheriting either static or eager base policies.

    Args:
        base: Base class for this policy. DynamicTFPolicyV2 or EagerTFPolicyV2.

    Returns:
        A TF Policy to be used with PPOTrainer.
    """
    class PPOTFPolicy(
        EntropyCoeffSchedule,
        LearningRateSchedule,
        KLCoeffMixin,
        ValueNetworkMixin,
        base
    ):
        def __init__(
            self,
            obs_space,
            action_space,
            config,
            existing_model=None,
            existing_inputs=None,
        ):
            validate_config(config)

            # Initialize base class.
            base.__init__(
                self,
                obs_space,
                action_space,
                config,
                existing_inputs=existing_inputs,
                existing_model=existing_model,
            )

            # Initialize MixIns.
            ValueNetworkMixin.__init__(self, config)
            KLCoeffMixin.__init__(self, config)
            EntropyCoeffSchedule.__init__(
                self, config["entropy_coeff"], config["entropy_coeff_schedule"]
            )
            LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

            # Note: this is a bit ugly, but loss and optimizer initialization must
            # happen after all the MixIns are initialized.
            self.maybe_initialize_optimizer_and_loss()

        @override(base)
        def loss(
            self,
            model: Union[ModelV2, "tf.keras.Model"],
            dist_class: Type[TFActionDistribution],
            train_batch: SampleBatch,
        ) -> Union[TensorType, List[TensorType]]:
            if isinstance(model, tf.keras.Model):
                logits, state, extra_outs = model(train_batch)
                value_fn_out = extra_outs[SampleBatch.VF_PREDS]
            else:
                logits, state = model(train_batch)
                value_fn_out = model.value_function()

            curr_action_dist = dist_class(logits, model)

            # RNN case: Mask away 0-padded chunks at end of time axis.
            if state:
                # Derive max_seq_len from the data itself, not from the seq_lens
                # tensor. This is in case e.g. seq_lens=[2, 3], but the data is still
                # 0-padded up to T=5 (as it's the case for attention nets).
                B = tf.shape(train_batch[SampleBatch.SEQ_LENS])[0]
                max_seq_len = tf.shape(logits)[0] // B

                mask = tf.sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
                mask = tf.reshape(mask, [-1])

                def reduce_mean_valid(t):
                    return tf.reduce_mean(tf.boolean_mask(t, mask))

            # non-RNN case: No masking.
            else:
                mask = None
                reduce_mean_valid = tf.reduce_mean

            prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

            logp_ratio = tf.exp(
                curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
                - train_batch[SampleBatch.ACTION_LOGP]
            )

            # Only calculate kl loss if necessary (kl-coeff > 0.0).
            if self.config["kl_coeff"] > 0.0:
                action_kl = prev_action_dist.kl(curr_action_dist)
                mean_kl_loss = reduce_mean_valid(action_kl)
            else:
                mean_kl_loss = tf.constant(0.0)

            curr_entropy = curr_action_dist.entropy()
            mean_entropy = reduce_mean_valid(curr_entropy)

            surrogate_loss = tf.minimum(
                train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
                train_batch[Postprocessing.ADVANTAGES]
                * tf.clip_by_value(
                    logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
                ),
            )
            mean_policy_loss = reduce_mean_valid(-surrogate_loss)

            # Compute a value function loss.
            if self.config["use_critic"]:
                vf_loss = tf.math.square(
                    value_fn_out - train_batch[Postprocessing.VALUE_TARGETS]
                )
                vf_loss_clipped = tf.clip_by_value(
                    vf_loss,
                    0,
                    self.config["vf_clip_param"],
                )
                mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
            # Ignore the value function.
            else:
                vf_loss_clipped = mean_vf_loss = tf.constant(0.0)

            total_loss = reduce_mean_valid(
                -surrogate_loss
                + self.config["vf_loss_coeff"] * vf_loss_clipped
                - self.entropy_coeff * curr_entropy
            )
            # Add mean_kl_loss (already processed through `reduce_mean_valid`),
            # if necessary.
            if self.config["kl_coeff"] > 0.0:
                total_loss += self.kl_coeff * mean_kl_loss

            # Store stats in policy for stats_fn.
            self._total_loss = total_loss
            self._mean_policy_loss = mean_policy_loss
            self._mean_vf_loss = mean_vf_loss
            self._mean_entropy = mean_entropy
            # Backward compatibility: Deprecate self._mean_kl.
            self._mean_kl_loss = self._mean_kl = mean_kl_loss
            self._value_fn_out = value_fn_out

            return total_loss

        @override(base)
        def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
        ):
            sample_batch = super().postprocess_trajectory(sample_batch)
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

        @override(base)
        def compute_gradients_fn(
            self, optimizer: LocalOptimizer, loss: TensorType
        ) -> ModelGradients:
            # Compute the gradients.
            variables = self.model.trainable_variables
            if isinstance(self.model, ModelV2):
                variables = variables()
            grads_and_vars = optimizer.compute_gradients(loss, variables)

            # Clip by global norm, if necessary.
            if self.config["grad_clip"] is not None:
                # Defuse inf gradients (due to super large losses).
                grads = [g for (g, v) in grads_and_vars]
                grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
                # If the global_norm is inf -> All grads will be NaN. Stabilize this
                # here by setting them to 0.0. This will simply ignore destructive loss
                # calculations.
                self.grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in grads]
                clipped_grads_and_vars = list(zip(self.grads, variables))
                return clipped_grads_and_vars
            else:
                return grads_and_vars

        @override(base)
        def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
            return {
                "cur_kl_coeff": tf.cast(self.kl_coeff, tf.float64),
                "cur_lr": tf.cast(self.cur_lr, tf.float64),
                "total_loss": self._total_loss,
                "policy_loss": self._mean_policy_loss,
                "vf_loss": self._mean_vf_loss,
                "vf_explained_var": explained_variance(
                    train_batch[Postprocessing.VALUE_TARGETS], self._value_fn_out
                ),
                "kl": self._mean_kl_loss,
                "entropy": self._mean_entropy,
                "entropy_coeff": tf.cast(self.entropy_coeff, tf.float64),
            }

    return PPOTFPolicy


PPODynamicTFPolicy = get_ppo_tf_policy(DynamicTFPolicyV2)
PPOEagerTFPolicy = get_ppo_tf_policy(EagerTFPolicyV2)
