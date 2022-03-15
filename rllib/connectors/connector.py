"""This file defines commonly used RLlib connectors.
"""

import abc
import logging
import tree  # pip install dm_tree
from typing import Dict, List, Tuple

import numpy as np

from ray.tune.registry import RLLIB_CONNECTOR, _global_registry
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import clip_action, unbatch, unsquash_action
from ray.rllib.utils.typing import Any, StateBatch, TensorStructType, TensorType

logger = logging.getLogger(__name__)

PolicyOutputType = Tuple[TensorStructType, StateBatch, dict]


class ConnectorContext():
    """Container class for data that may be needed for running connectors.
    """

    # TODO(jungong) : figure out how to do this in a remote setting after
    # policy API is significantly simplified.
    # TODO(jungong) : we should be careful and opinionated about the things
    # folks have access to in the context.
    # For example, we should put view_requirements and action_space here,
    # instead of the raw policy, which may not be available during inference.

    def __init__(self, policy=None):
        self.policy = policy


class Connector(abc.ABC):
    def __init__(self):
        # This gets flipped to False in PolicyRunner for inference.
        self.is_training = True

    def for_training(self, is_training: bool):
        self.is_training = is_training

    # The only API we enforce is that Connector have to be serializable.
    def to_config(self):
        # Must implement by each connector.
        return NotImplementedError


class AgentConnector(Connector):
    def reset(self, env_id: str):
        pass

    def on_policy_output(self, env_id: str, agent_id: str, output: PolicyOutputType):
        # May be useful for certain connector (e.g., state_out buffering connect).
        pass

    def __call__(self, ctx: ConnectorContext, env_id: str, agent_id: str, d: Any):
        raise NotImplementedError


class ActionConnector(Connector):
    def __call__(self, ctx: ConnectorContext, d: PolicyOutputType):
        raise NotImplementedError


def register_connector(name: str, cls: Connector):
    """Register a connector for use with RLlib.

    Args:
        name: Name to register.
        cls: Callable that creates an env.
    """
    if not issubclass(cls, Connector):
        raise TypeError("Can only register Connector type.", cls)
    _global_registry.register(RLLIB_CONNECTOR, name, cls)


def get_connector(name, params) -> Connector:
    if not _global_registry.contains(RLLIB_CONNECTOR, name):
        raise NameError("connector not found.", name)
    cls = _global_registry.get(RLLIB_CONNECTOR, name)
    return cls.from_config(params)
