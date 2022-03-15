from ray.rllib.connectors.connector import (
    ConnectorContext,
    PolicyOutputType,
    ActionConnector,
    register_connector,
    get_connector,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import clip_action, unsquash_action, unbatch


class ConvertToNumpyConnector(ActionConnector):
    def __call__(self, ctx: ConnectorContext, d: PolicyOutputType):
        actions, states, fetches = d
        return convert_to_numpy(actions), convert_to_numpy(states), fetches

    def to_config(self):
        return ConvertToNumpyConnector.__name__, None

    @staticmethod
    def from_config(_):
        return ConvertToNumpyConnector()


register_connector(ConvertToNumpyConnector.__name__, ConvertToNumpyConnector)


# TODO(jungong) : figure out if we need to unbatch actions for a single agent.
class UnbatchActionsConnector(ActionConnector):
    """Split action-component batches into single action rows.
    """
    def __call__(self, ctx: ConnectorContext, d: PolicyOutputType):
        actions, states, fetches = d
        return unbatch(actions), states, fetches

    def to_config(self):
        return UnbatchActionsConnector.__name__, None

    @staticmethod
    def from_config(_):
        return UnbatchActionsConnector()


register_connector(UnbatchActionsConnector.__name__, UnbatchActionsConnector)


class NormalizeActionsConnector(ActionConnector):
    def __call__(self, ctx: ConnectorContext, d: PolicyOutputType):
        actions, states, fetches = d
        actions = unsquash_action(actions, ctx.policy.action_space_struct)
        return actions, states, fetches

    def to_config(self):
        return NormalizeActionsConnector.__name__, None

    @staticmethod
    def from_config(_):
        return NormalizeActionsConnector()


register_connector(NormalizeActionsConnector.__name__, NormalizeActionsConnector)


class ClipActionsConnector(ActionConnector):
    def __call__(self, ctx: ConnectorContext, d: PolicyOutputType):
        actions, states, fetches = d
        actions = clip_action(action, ctx.policy.action_space_struct)
        return actions, states, fetches

    def to_config(self):
        return ClipActionsConnector.__name__, None

    @staticmethod
    def from_config(_):
        return ClipActionsConnector()


register_connector(ClipActionsConnector.__name__, ClipActionsConnector)


class ActionConnectorPipeline(ActionConnector):
    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors

    def for_training(self, is_training: bool):
        self.is_training = is_training
        for c in self.connectors:
            c.for_training(is_training)

    def __call__(self, ctx: ConnectorContext, d: PolicyOutputType):
        for c in self.connectors:
            d = c(ctx, d)
        return d

    def to_config(self):
        return ActionConnectorPipeline.__name__, [c.to_config() for c in self.connectors]

    @staticmethod
    def from_config(params):
        assert type(params) == list, "ActionConnectorPipeline takes a list of connector params."
        connectors = [get_connector(name, subparams) for name, subparams in params]
        return ActionConnectorPipeline(connectors)


register_connector(ActionConnectorPipeline.__name__, ActionConnectorPipeline)
