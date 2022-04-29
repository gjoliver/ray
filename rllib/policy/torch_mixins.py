from typing import Dict, List, Union

from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import (
    TensorType,
)

torch, nn = try_import_torch()

# TODO: (sven) Unify hyperparam annealing procedures across RLlib (tf/torch)
#   and for all possible hyperparams, not just lr.
@DeveloperAPI
class LearningRateSchedule:
    """Mixin for TorchPolicy that adds a learning rate schedule."""

    @DeveloperAPI
    def __init__(self, lr, lr_schedule):
        self._lr_schedule = None
        if lr_schedule is None:
            self.cur_lr = lr
        else:
            self._lr_schedule = PiecewiseSchedule(
                lr_schedule, outside_value=lr_schedule[-1][-1], framework=None
            )
            self.cur_lr = self._lr_schedule.value(0)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p["lr"] = self.cur_lr


@DeveloperAPI
class EntropyCoeffSchedule:
    """Mixin for TorchPolicy that adds entropy coeff decay."""

    @DeveloperAPI
    def __init__(self, entropy_coeff, entropy_coeff_schedule):
        self._entropy_coeff_schedule = None
        if entropy_coeff_schedule is None:
            self.entropy_coeff = entropy_coeff
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
            self.entropy_coeff = self._entropy_coeff_schedule.value(0)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(EntropyCoeffSchedule, self).on_global_var_update(global_vars)
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"]
            )


class KLCoeffMixin:
    """Assigns the `update_kl()` method to the PPOPolicy.

    This is used in PPO's execution plan (see ppo.py) for updating the KL
    coefficient after each learning step based on `config.kl_target` and
    the measured KL value (from the train_batch).
    """

    def __init__(self, config):
        # The current KL value (as python float).
        self.kl_coeff = config["kl_coeff"]
        # Constant target value.
        self.kl_target = config["kl_target"]

    def update_kl(self, sampled_kl):
        # Update the current KL value based on the recently measured value.
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        # Return the current KL value.
        return self.kl_coeff

    @override(TorchPolicy)
    def get_state(self) -> Union[Dict[str, TensorType], List[TensorType]]:
        state = super().get_state()
        # Add current kl-coeff value.
        state["current_kl_coeff"] = self.kl_coeff
        return state

    @override(TorchPolicy)
    def set_state(self, state: dict) -> None:
        # Set current kl-coeff value first.
        self.kl_coeff = state.pop("current_kl_coeff", self.config["kl_coeff"])
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

            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0].item()

        # When not doing GAE, we do not require the value function's output.
        else:

            def value(*args, **kwargs):
                return 0.0

        self._value = value

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        """Defines extra fetches per action computation.

        Args:
            input_dict (Dict[str, TensorType]): The input dict used for the action
                computing forward pass.
            state_batches (List[TensorType]): List of state tensors (empty for
                non-RNNs).
            model (ModelV2): The Model object of the Policy.
            action_dist (TorchDistributionWrapper): The instantiated distribution
                object, resulting from the model's outputs and the given
                distribution class.

        Returns:
            Dict[str, TensorType]: Dict with extra tf fetches to perform per
                action computation.
        """
        # Return value function outputs. VF estimates will hence be added to
        # the SampleBatches produced by the sampler(s) to generate the train
        # batches going into the loss function.
        return {
            SampleBatch.VF_PREDS: model.value_function(),
        }


class ComputeGAEMixIn:
    """Postprocess SampleBatch to Compute GAE before they get used for training.
    """
    def __init__(self):
        pass

    @DeveloperAPI
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )