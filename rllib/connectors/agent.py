from collections import defaultdict, namedtuple
import numpy as np
import tree

from ray.rllib.connectors.connector import (
    ConnectorContext,
    AgentConnector,
    PolicyOutputType,
    register_connector,
    get_connector,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import AgentConnectorsOut, Any, TrainerConfigDict


class AgentConnectorPipeline(AgentConnector):
    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors

    def for_training(self, is_training: bool):
        self.is_training = is_training
        for c in self.connectors:
            c.for_training(is_training)

    def reset(self, env_id: str):
        for c in self.connectors:
            c.reset(env_id)

    def on_policy_output(self, env_id: str, agent_id: str, output: PolicyOutputType):
        for c in self.connectors:
            c.on_policy_output(env_id, agent_id, output)

    def __call__(self, ctx: ConnectorContext, env_id: str, agent_id: str, d: Any):
        for c in self.connectors:
            d = c(ctx, env_id, agent_id, d)
        return d

    def to_config(self):
        return AgentConnectorPipeline.__name__, [c.to_config() for c in self.connectors]

    @staticmethod
    def from_config(params):
        assert type(params) == list, "AgentConnectorPipeline takes a list of connector params."
        connectors = [get_connector(name, subparams) for name, subparams in params]
        return AgentConnectorPipeline(connectors)


register_connector(AgentConnectorPipeline.__name__, AgentConnectorPipeline)


# TODO(jungong) : figure out when do we need to tree.flatten() env obs, etc.
class FlattenEnvDataConnector(AgentConnector):
    def __call__(self, ctx: ConnectorContext, env_id: str, agent_id: str, d: Any):
        assert type(d) == dict, "data param must be of type Dict[str, TensorStructType]"

        flattened = {}
        for k, v in d.items():
            if k in [SampleBatch.INFOS, SampleBatch.ACTIONS] or k.startswith("state_out_"):
                # Do not flatten infos, actions, and state_out_ columns.
                flattened[k] = v
                continue
            if v is None:
                # Keep the same column shape.
                flattened[k] = None
                continue
            flattened[k] = np.array(tree.flatten(v))
        return flattened

    def to_config(self):
        return FlattenEnvDataConnector.__name__, None

    @staticmethod
    def from_config(_):
        return FlattenEnvDataConnector()


register_connector(FlattenEnvDataConnector.__name__, FlattenEnvDataConnector)


class ClipRewardConnector(AgentConnector):
    def __init__(self, sign=False, limit=None):
        super().__init__()
        assert sign == False or limit is None, "should not enable both sign and limit reward clipping."
        self.sign = sign
        self.limit = limit

    def __call__(self, ctx: ConnectorContext, env_id: str, agent_id: str, d: Any):
        assert type(d) == dict, "data param must be of type Dict[str, TensorStructType]"

        assert SampleBatch.REWARDS in d, "input data does not have reward column."
        if self.sign:
            d[SampleBatch.REWARDS] = np.sign(d[SampleBatch.REWARDS])
        elif self.limit:
            d[SampleBatch.REWARDS] = np.clip(
                dS[ampleBatch.REWARDS], a_min=-self.limit, a_max=self.limit,
            )
        return d

    def to_config(self):
        return ClipRewardConnector.__name__, {"sign": self.sign, "limit": self.limit}

    @staticmethod
    def from_config(params):
        return ClipRewardConnector(**params)


register_connector(ClipRewardConnector.__name__, ClipRewardConnector)


class AgentState(object):
    def __init__(self):
        self.t = 0
        self.action = None
        self.states = None


class StateBufferConnector(AgentConnector):
    def __init__(self):
        super().__init__()
        self._states = defaultdict(lambda: defaultdict(AgentState))

    def reset(self, env_id: str):
        del self._states[env_id]

    def on_policy_output(self, env_id: str, agent_id: str, output: PolicyOutputType):
        # Buffer latest output states for next input __call__.
        action, states, _ = output
        agent_state = self._states[env_id][agent_id]
        agent_state.action = convert_to_numpy(action)
        agent_state.states = convert_to_numpy(states)

    def __call__(self, ctx: ConnectorContext, env_id: str, agent_id: str, d: Any):
        agent_state = self._states[env_id][agent_id]

        d.update({
            SampleBatch.T: agent_state.t,
            SampleBatch.ENV_ID: env_id,
        })

        if agent_state.states is not None:
            states = agent_state.states
        else:
            states = ctx.policy.get_initial_state()
        for i, v in enumerate(states):
            d["state_out_{}".format(i)] = v

        if agent_state.action is not None:
            d[SampleBatch.ACTIONS] = agent_state.action  # Last action
        else:
            # Default zero action.
            d[SampleBatch.ACTIONS] = tree.map_structure(
                lambda s: np.zeros_like(s.sample(), s.dtype)
                if hasattr(s, "dtype")
                else np.zeros_like(s.sample()),
                ctx.policy.action_space_struct,
            )

        agent_state.t += 1

        return d

    def to_config(self):
        return StateBufferConnector.__name__, None

    @staticmethod
    def from_config(params):
        return StateBufferConnector()


register_connector(StateBufferConnector.__name__, StateBufferConnector)


class ViewRequirementConnector(AgentConnector):
    """This connector does 2 things:
    1. It filters data columns based on view_requirements for training and inference.
    2. It buffers the right amount of history for computing the sample batch for
       action computation.
    The output of this connector is AgentConnectorsOut, which basically is
    a tuple of 2 things:
    {
        "for_training": {"obs": ...}
        "for_action": SampleBatch
    }
    The "for_training" dict, which contains data for the latest time slice,
    can be used to construct a complete episode by SampleCollecotr for training purpose.
    The "for_action" SampleBatch can be used to directly call the policy.
    """
    def __init__(self):
        super().__init__()
        self._agent_data = defaultdict(lambda: defaultdict(SampleBatch))

    def reset(self, env_id: str):
        if env_id in self._agent_data:
            del self._agent_data[env_id]

    def _get_sample_batch_for_action(self, view_requirements, agent_batch) -> SampleBatch:
        # TODO(jungong) : actually support buildling input sample batch with all the
        #  view shift requirements, etc.
        # For now, we use some simple logics for demo purpose.
        input_batch = SampleBatch()
        for k, v in view_requirements.items():
            if not v.used_for_compute_actions:
                continue
            data_col = v.data_col or k
            if data_col not in agent_batch:
                continue
            input_batch[k] = agent_batch[data_col][-1:]
        input_batch.count = 1
        return input_batch

    def __call__(
        self, ctx: ConnectorContext, env_id: str, agent_id: str, d: Any
    ) -> AgentConnectorsOut:
        if not ctx.policy:
            return d
        vr = ctx.policy.view_requirements

        training_dict = {}
        # We construct a proper per-timeslice dict in training mode,
        # for Sampler to construct a complete episode for back propagation.
        if self.is_training:
            # Filter columns that are not needed for traing.
            for col, req in vr.items():
                # Not used for training.
                if not req.used_for_training:
                    continue

                # Create the batch of data from the different buffers.
                data_col = req.data_col or col
                if data_col not in d:
                    continue

                training_dict[data_col] = d[data_col]

        # Agent batch is our buffer of necessary history for computing
        # the SampleBatches for policy forward pass.
        # This is used by both training and inference.
        agent_batch = self._agent_data[env_id][agent_id]
        for col, req in vr.items():
            # Not used for action computation.
            if not req.used_for_compute_actions:
                continue

            # Create the batch of data from the different buffers.
            data_col = req.data_col or col
            if data_col not in d:
                continue

            # Add batch dim to this data_col.
            d_col = np.expand_dims(d[data_col], axis=0)

            if col in agent_batch:
                # Stack along batch dim.
                agent_batch[data_col] = np.vstack((agent_batch[data_col], d_col))
            else:
                agent_batch[data_col] = d_col
            # Only keep the useful part of the history.
            h = req.shift_from if req.shift_from else -1
            assert h <= 0, "Can use future data to compute action"
            agent_batch[data_col] = agent_batch[data_col][h:]

        sample_batch = self._get_sample_batch_for_action(vr, agent_batch)

        return AgentConnectorsOut(training_dict, sample_batch)

    def to_config(self):
        return ViewRequirementConnector.__name__, None

    @staticmethod
    def from_config(params):
        return ViewRequirementConnector()


register_connector(ViewRequirementConnector.__name__, ViewRequirementConnector)


# TODO(junogng): figure out if we really need this.
# The idea is to provide defaults to make things backward compatible.
def get_default_global_env_connectors(config: TrainerConfigDict) -> AgentConnectorPipeline:
    global_connectors = [FlattenEnvDataConnector()]

    if config["clip_rewards"] is True:
        global_connectors.append(ClipRewardConnector(sign=True))
    elif type(config["clip_rewards"]) == float:
        global_connectors.append(ClipRewardConnector(limit=abs(config["clip_rewards"])))

    return AgentConnectorPipeline(global_connectors)
