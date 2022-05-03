from typing import Dict, List, Union

from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_tf, get_variable
from ray.rllib.utils.tf_utils import make_tf_callable
from ray.rllib.utils.typing import (
    LocalOptimizer,
    ModelGradients,
    TensorType,
)

tf1, tf, tfv = try_import_tf()


@DeveloperAPI
class LearningRateSchedule:
    """Mixin for TFPolicy that adds a learning rate schedule."""

    @DeveloperAPI
    def __init__(self, lr, lr_schedule):
        self._lr_schedule = None
        if lr_schedule is None:
            self.cur_lr = tf1.get_variable("lr", initializer=lr, trainable=False)
        else:
            self._lr_schedule = PiecewiseSchedule(
                lr_schedule, outside_value=lr_schedule[-1][-1], framework=None
            )
            self.cur_lr = tf1.get_variable(
                "lr", initializer=self._lr_schedule.value(0), trainable=False
            )
            if self.framework == "tf":
                self._lr_placeholder = tf1.placeholder(dtype=tf.float32, name="lr")
                self._lr_update = self.cur_lr.assign(
                    self._lr_placeholder, read_value=False
                )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule is not None:
            new_val = self._lr_schedule.value(global_vars["timestep"])
            if self.framework == "tf":
                self.get_session().run(
                    self._lr_update, feed_dict={self._lr_placeholder: new_val}
                )
            else:
                self.cur_lr.assign(new_val, read_value=False)
                # This property (self._optimizer) is (still) accessible for
                # both TFPolicy and any TFPolicy_eager.
                self._optimizer.learning_rate.assign(self.cur_lr)

    @override(TFPolicy)
    def optimizer(self):
        return tf1.train.AdamOptimizer(learning_rate=self.cur_lr)


@DeveloperAPI
class EntropyCoeffSchedule:
    """Mixin for TFPolicy that adds entropy coeff decay."""

    @DeveloperAPI
    def __init__(self, entropy_coeff, entropy_coeff_schedule):
        self._entropy_coeff_schedule = None
        if entropy_coeff_schedule is None:
            self.entropy_coeff = get_variable(
                entropy_coeff, framework="tf", tf_name="entropy_coeff", trainable=False
            )
        else:
            # Allows for custom schedule similar to lr_schedule format
            if isinstance(entropy_coeff_schedule, list):
                self._entropy_coeff_schedule = PiecewiseSchedule(
                    entropy_coeff_schedule,
                    outside_value=entropy_coeff_schedule[-1][-1],
                    framework=None,
                )
            else:
                # Implements previous version but enforces outside_value
                self._entropy_coeff_schedule = PiecewiseSchedule(
                    [[0, entropy_coeff], [entropy_coeff_schedule, 0.0]],
                    outside_value=0.0,
                    framework=None,
                )

            self.entropy_coeff = get_variable(
                self._entropy_coeff_schedule.value(0),
                framework="tf",
                tf_name="entropy_coeff",
                trainable=False,
            )
            if self.framework == "tf":
                self._entropy_coeff_placeholder = tf1.placeholder(
                    dtype=tf.float32, name="entropy_coeff"
                )
                self._entropy_coeff_update = self.entropy_coeff.assign(
                    self._entropy_coeff_placeholder, read_value=False
                )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._entropy_coeff_schedule is not None:
            new_val = self._entropy_coeff_schedule.value(global_vars["timestep"])
            if self.framework == "tf":
                self.get_session().run(
                    self._entropy_coeff_update,
                    feed_dict={self._entropy_coeff_placeholder: new_val},
                )
            else:
                self.entropy_coeff.assign(new_val, read_value=False)


class KLCoeffMixin:
    """Assigns the `update_kl()` and other KL-related methods to the PPOPolicy.

    This is used in PPO's execution plan (see ppo.py) for updating the KL
    coefficient after each learning step based on `config.kl_target` and
    the measured KL value (from the train_batch).
    """

    def __init__(self, config):
        # The current KL value (as python float).
        self.kl_coeff_val = config["kl_coeff"]
        # The current KL value (as tf Variable for in-graph operations).
        self.kl_coeff = get_variable(
            float(self.kl_coeff_val),
            tf_name="kl_coeff",
            trainable=False,
            framework=config["framework"],
        )
        # Constant target value.
        self.kl_target = config["kl_target"]
        if self.framework == "tf":
            self._kl_coeff_placeholder = tf1.placeholder(
                dtype=tf.float32, name="kl_coeff"
            )
            self._kl_coeff_update = self.kl_coeff.assign(
                self._kl_coeff_placeholder, read_value=False
            )

    def update_kl(self, sampled_kl):
        # Update the current KL value based on the recently measured value.
        # Increase.
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff_val *= 1.5
        # Decrease.
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff_val *= 0.5
        # No change.
        else:
            return self.kl_coeff_val

        # Make sure, new value is also stored in graph/tf variable.
        self._set_kl_coeff(self.kl_coeff_val)

        # Return the current KL value.
        return self.kl_coeff_val

    def _set_kl_coeff(self, new_kl_coeff):
        # Set the (off graph) value.
        self.kl_coeff_val = new_kl_coeff

        # Update the tf/tf2 Variable (via session call for tf or `assign`).
        if self.framework == "tf":
            self.get_session().run(
                self._kl_coeff_update,
                feed_dict={self._kl_coeff_placeholder: self.kl_coeff_val},
            )
        else:
            self.kl_coeff.assign(self.kl_coeff_val, read_value=False)

    @override(Policy)
    def get_state(self) -> Union[Dict[str, TensorType], List[TensorType]]:
        state = super().get_state()
        # Add current kl-coeff value.
        state["current_kl_coeff"] = self.kl_coeff_val
        return state

    @override(Policy)
    def set_state(self, state: dict) -> None:
        # Set current kl-coeff value first.
        self._set_kl_coeff(state.pop("current_kl_coeff", self.config["kl_coeff"]))
        # Call super's set_state with rest of the state dict.
        super().set_state(state)


class ValueNetworkMixin:
    """Assigns the `_value()` method to the PPOPolicy.

    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    """
    def __init__(self, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:

            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            @make_tf_callable(self.get_session())
            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                if isinstance(self.model, tf.keras.Model):
                    _, _, extra_outs = self.model(input_dict)
                    return extra_outs[SampleBatch.VF_PREDS][0]
                else:
                    model_out, _ = self.model(input_dict)
                    # [0] = remove the batch dim.
                    return self.model.value_function()[0]

        # When not doing GAE, we do not require the value function's output.
        else:

            @make_tf_callable(self.get_session())
            def value(*args, **kwargs):
                return tf.constant(0.0)

        self._value = value

    def extra_action_out_fn(self) -> Dict[str, TensorType]:
        # TODO: (sven) Deprecate once we only allow native keras models.
        fetches = super().extra_action_out_fn()
        # Keras models return values for each call in third return argument
        # (dict).
        if isinstance(self.model, tf.keras.Model):
            return fetches
        # Return value function outputs. VF estimates will hence be added to the
        # SampleBatches produced by the sampler(s) to generate the train batches
        # going into the loss function.
        fetches.update({
            SampleBatch.VF_PREDS: self.model.value_function(),
        })
        return fetches


class ComputeGAEMixIn:
    """Postprocess SampleBatch to Compute GAE before they get used for training.
    """
    def __init__(self):
        pass

    @DeveloperAPI
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        sample_batch = super().postprocess_trajectory(sample_batch)
        return compute_gae_for_sample_batch(
            self, sample_batch, other_agent_batches, episode
        )


class ComputeAndClipGradsMixIn:
    """Compute and maybe clip gradients.
    """
    def __init__(self):
        pass

    @DeveloperAPI
    def compute_gradients_fn(
        self,
        optimizer: LocalOptimizer,
        loss: TensorType
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
