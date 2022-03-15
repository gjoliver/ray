from ray.rllib.connectors.connector import (
    Connector,
    get_connector,
)
from ray.rllib.connectors import action
from ray.rllib.connectors import agent
from ray.rllib.utils.typing import Any, StateBatch, TensorStructType, TensorType, TrainerConfigDict
from typing import Dict


def get_connectors_from_cfg(config: dict) -> Dict[str, Connector]:
    return {
        k: get_connector(*v) for k, v in config.items()
    }
