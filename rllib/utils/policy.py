import gym
import pickle
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import (
    ActionConnectorDataType,
    AgentConnectorDataType,
    AgentConnectorsOutput,
    PartialAlgorithmConfigDict,
    PolicyID,
    PolicyState,
    TensorStructType,
)
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.rllib.policy.policy import Policy

tf1, tf, tfv = try_import_tf()


def enable_eager_execution_if_tf2():
    """Helper function for enalbing TF2 eager exec when restoring a TF2 policy.

    Note: this is done as a separate util so users can run this first thing
    after program starts.
    """
    if tf1 and not tf1.executing_eagerly():
        tf1.enable_eager_execution()


@PublicAPI
def create_policy_for_framework(
    policy_id: str,
    policy_class: "Policy",
    merged_config: PartialAlgorithmConfigDict,
    observation_space: gym.Space,
    action_space: gym.Space,
    worker_index: int = 0,
    session_creator: Optional[Callable[[], "tf1.Session"]] = None,
    seed: Optional[int] = None,
):
    """Frame specific policy creation logics.

    Args:
        policy_id: Policy ID.
        policy_class: Policy class type.
        merged_config: Complete policy config.
        observation_space: Observation space of env.
        action_space: Action space of env.
        worker_index: Index of worker holding this policy. Default is 0.
        session_creator: An optional tf1.Session creation callable.
        seed: Optional random seed.
    """
    framework = merged_config.get("framework", "tf")
    # Tf.
    if framework in ["tf2", "tf", "tfe"]:
        var_scope = policy_id + (f"_wk{worker_index}" if worker_index else "")
        # For tf static graph, build every policy in its own graph
        # and create a new session for it.
        if framework == "tf":
            with tf1.Graph().as_default():
                if session_creator:
                    sess = session_creator()
                else:
                    sess = tf1.Session(
                        config=tf1.ConfigProto(
                            gpu_options=tf1.GPUOptions(allow_growth=True)
                        )
                    )
                with sess.as_default():
                    # Set graph-level seed.
                    if seed is not None:
                        tf1.set_random_seed(seed)
                    with tf1.variable_scope(var_scope):
                        return policy_class(
                            observation_space, action_space, merged_config
                        )
        # For tf-eager: no graph, no session.
        else:
            with tf1.variable_scope(var_scope):
                return policy_class(observation_space, action_space, merged_config)
    # Non-tf: No graph, no session.
    else:
        return policy_class(observation_space, action_space, merged_config)


@PublicAPI(stability="alpha")
def parse_policy_specs_from_checkpoint(
    path: str,
) -> Tuple[PartialAlgorithmConfigDict, Dict[str, PolicySpec], Dict[str, PolicyState]]:
    """Read and parse policy specifications from a checkpoint file.

    Args:
        path: Path to a policy checkpoint.

    Returns:
        A tuple of: base policy config, dictionary of policy specs, and
        dictionary of policy states.
    """
    with open(path, "rb") as f:
        checkpoint_dict = pickle.load(f)
    # Policy data is contained as a serialized binary blob under their
    # ID keys.
    w = pickle.loads(checkpoint_dict["worker"])

    policy_config = w["policy_config"]
    assert policy_config.get("enable_connectors", False), (
        "load_policies_from_checkpoint only works for checkpoints generated by stacks "
        "with connectors enabled."
    )
    policy_states = w["state"]
    serialized_policy_specs = w["policy_specs"]
    policy_specs = {
        id: PolicySpec.deserialize(spec) for id, spec in serialized_policy_specs.items()
    }

    return policy_config, policy_specs, policy_states


@PublicAPI(stability="alpha")
def load_policies_from_checkpoint(
    path: str, policy_ids: Optional[List[PolicyID]] = None
) -> Dict[str, "Policy"]:
    """Load the list of policies from a connector enabled policy checkpoint.

    Args:
        path: File path to the checkpoint file.
        policy_ids: a list of policy IDs to be restored. If missing, we will
        load all policies contained in this checkpoint.

    Returns:

    """
    policy_config, policy_specs, policy_states = parse_policy_specs_from_checkpoint(
        path
    )

    policies = {}
    for id, policy_spec in policy_specs.items():
        if policy_ids and id not in policy_ids:
            # User want specific policies, and this is not one of them.
            continue

        merged_config = merge_dicts(policy_config, policy_spec.config or {})
        policy = create_policy_for_framework(
            id,
            policy_spec.policy_class,
            merged_config,
            policy_spec.observation_space,
            policy_spec.action_space,
        )
        if id in policy_states:
            policy.set_state(policy_states[id])
        if policy.agent_connectors:
            policy.agent_connectors.is_training(False)
        policies[id] = policy

    return policies


@PublicAPI(stability="alpha")
def local_policy_inference(
    policy: "Policy",
    env_id: str,
    agent_id: str,
    obs: TensorStructType,
) -> TensorStructType:
    """Run a connector enabled policy using environment observation.

    policy_inference manages policy and agent/action connectors,
    so the user does not have to care about RNN state buffering or
    extra fetch dictionaries.
    Note that connectors are intentionally run separately from
    compute_actions_from_input_dict(), so we can have the option
    of running per-user connectors on the client side in a
    server-client deployment.

    Args:
        policy: Policy.
        env_id: Environment ID.
        agent_id: Agent ID.
        obs: Env obseration.

    Returns:
        List of outputs from policy forward pass.
    """
    assert (
        policy.agent_connectors
    ), "policy_inference only works with connector enabled policies."

    # TODO(jungong) : support multiple env, multiple agent inference.
    input_dict = {SampleBatch.NEXT_OBS: obs}
    acd_list: List[AgentConnectorDataType] = [
        AgentConnectorDataType(env_id, agent_id, input_dict)
    ]
    ac_outputs: List[AgentConnectorsOutput] = policy.agent_connectors(acd_list)
    outputs = []
    for ac in ac_outputs:
        policy_output = policy.compute_actions_from_input_dict(ac.data.for_action)

        if policy.action_connectors:
            acd = ActionConnectorDataType(env_id, agent_id, policy_output)
            acd = policy.action_connectors(acd)
            actions = acd.output
        else:
            actions = policy_output[0]

        outputs.append(actions)

        # Notify agent connectors with this new policy output.
        # Necessary for state buffering agent connectors, for example.
        policy.agent_connectors.on_policy_output(
            ActionConnectorDataType(env_id, agent_id, policy_output)
        )
    return outputs
