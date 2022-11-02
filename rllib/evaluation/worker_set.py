import gym
import logging
import importlib.util
import os
from typing import (
    Callable,
    Container,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import ray
from ray.actor import ActorHandle
from ray.exceptions import RayError, RayActorError
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import (
    Deprecated,
    deprecation_warning,
    DEPRECATED_VALUE,
)
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.typing import (
    AgentID,
    AlgorithmConfigDict,
    EnvCreator,
    EnvType,
    EpisodeID,
    PartialAlgorithmConfigDict,
    PolicyID,
    SampleBatchType,
    TensorType,
)
from ray.util.debug import log_once

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)

# Generic type var for foreach_* methods.
T = TypeVar("T")


@DeveloperAPI
class WorkerSet:
    """Set of RolloutWorkers with n @ray.remote workers and zero or one local worker.

    Where: n >= 0.
    """

    def __init__(
        self,
        *,
        env_creator: Optional[EnvCreator] = None,
        validate_env: Optional[Callable[[EnvType], None]] = None,
        default_policy_class: Optional[Type[Policy]] = None,
        config: Optional[Union["AlgorithmConfig", AlgorithmConfigDict]] = None,
        num_workers: int = 0,
        local_worker: bool = True,
        logdir: Optional[str] = None,
        _setup: bool = True,
        # deprecated args.
        policy_class=DEPRECATED_VALUE,
        trainer_config=DEPRECATED_VALUE,
    ):
        """Initializes a WorkerSet instance.

        Args:
            env_creator: Function that returns env given env config.
            validate_env: Optional callable to validate the generated
                environment (only on worker=0).
            default_policy_class: An optional default Policy class to use inside
                the (multi-agent) `policies` dict. In case the PolicySpecs in there
                have no class defined, use this `default_policy_class`.
                If None, PolicySpecs will be using the Algorithm's default Policy
                class.
            config: Optional AlgorithmConfig (or config dict).
            num_workers: Number of remote rollout workers to create.
            local_worker: Whether to create a local (non @ray.remote) worker
                in the returned set as well (default: True). If `num_workers`
                is 0, always create a local worker.
            logdir: Optional logging directory for workers.
            _setup: Whether to setup workers. This is only for testing.
        """
        if policy_class != DEPRECATED_VALUE:
            deprecation_warning(
                old="WorkerSet(policy_class=..)",
                new="WorkerSet(default_policy_class=..)",
                error=False,
            )
            default_policy_class = policy_class
        if trainer_config != DEPRECATED_VALUE:
            deprecation_warning(
                old="WorkerSet(trainer_config=..)",
                new="WorkerSet(config=..)",
                error=False,
            )
            config = trainer_config

        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

        # Make sure `config` is an AlgorithmConfig object.
        if not config:
            config = AlgorithmConfig()
        elif isinstance(config, dict):
            config = AlgorithmConfig.from_dict(config)

        self._env_creator = env_creator
        self._policy_class = default_policy_class
        self._remote_config = config
        self._remote_args = {
            "num_cpus": self._remote_config.num_cpus_per_worker,
            "num_gpus": self._remote_config.num_gpus_per_worker,
            "resources": self._remote_config.custom_resources_per_worker,
        }
        self._cls = RolloutWorker.as_remote(**self._remote_args).remote
        self._logdir = logdir

        if _setup:
            try:
                self._setup(
                    validate_env=validate_env,
                    config=config,
                    num_workers=num_workers,
                    local_worker=local_worker,
                )
            # WorkerSet creation possibly fails, if some (remote) workers cannot
            # be initialized properly (due to some errors in the RolloutWorker's
            # constructor).
            except RayActorError as e:
                # In case of an actor (remote worker) init failure, the remote worker
                # may still exist and will be accessible, however, e.g. calling
                # its `sample.remote()` would result in strange "property not found"
                # errors.
                if e.actor_init_failed:
                    # Raise the original error here that the RolloutWorker raised
                    # during its construction process. This is to enforce transparency
                    # for the user (better to understand the real reason behind the
                    # failure).
                    # - e.args[0]: The RayTaskError (inside the caught RayActorError).
                    # - e.args[0].args[2]: The original Exception (e.g. a ValueError due
                    # to a config mismatch) thrown inside the actor.
                    raise e.args[0].args[2]
                # In any other case, raise the RayActorError as-is.
                else:
                    raise e

    def _setup(
        self,
        *,
        validate_env: Optional[Callable[[EnvType], None]] = None,
        config: Optional[AlgorithmConfigDict] = None,
        num_workers: int = 0,
        local_worker: bool = True,
    ):
        """Initializes a WorkerSet instance.
        Args:
            validate_env: Optional callable to validate the generated
                environment (only on worker=0).
            trainer_config: Optional dict that extends the common config of
                the Algorithm class.
            num_workers: Number of remote rollout workers to create.
            local_worker: Whether to create a local (non @ray.remote) worker
                in the returned set as well (default: True). If `num_workers`
                is 0, always create a local worker.
        """
        # Force a local worker if num_workers == 0 (no remote workers).
        # Otherwise, this WorkerSet would be empty.
        self._local_worker = None
        if num_workers == 0:
            local_worker = True
        self._local_config = config.copy(copy_frozen=False).framework(
            tf_session_args=config.local_tf_session_args
        )

        if config.input_ == "dataset":
            # Create the set of dataset readers to be shared by all the
            # rollout workers.
            self._ds, self._ds_shards = get_dataset_and_shards(config, num_workers)
        else:
            self._ds = None
            self._ds_shards = None

        # Create a number of @ray.remote workers.
        self.__worker_manager = FaultTolerantActorManager()
        self.add_workers(
            num_workers,
            validate=config.validate_workers_after_construction,
        )

        # If num_workers > 0 and we don't have an env on the local worker,
        # get the observation- and action spaces for each policy from
        # the first remote worker (which does have an env).
        if (
            local_worker
            and self.__worker_manager.num_actors() > 0
            and not config.create_env_on_local_worker
            and (not config.observation_space or not config.action_space)
        ):
            spaces = self._get_spaces_from_remote_worker()
        else:
            spaces = None

        # Create a local worker, if needed.
        if local_worker:
            self._local_worker = self._make_worker(
                cls=RolloutWorker,
                env_creator=self._env_creator,
                validate_env=validate_env,
                worker_index=0,
                num_workers=num_workers,
                config=self._local_config,
                spaces=spaces,
            )

    def _get_spaces_from_remote_worker(self):
        """Guess observation and action spaces from a remote worker.

        Returns:
            A dict mapping from policy ids to spaces.
        """
        # Try to figure out spaces from the first remote worker.
        remote_spaces = self.foreach_worker(
            lambda worker: worker.foreach_policy(
                lambda p, pid: (pid, p.observation_space, p.action_space)
            ),
            remote_worker_indices=[0],
            local_worker=False,
        )
        if not remote_spaces:
            raise ValueError(
                "Could not get observation and action spaces from remote "
                "worker. Maybe specify them manually in the config?"
            )
        spaces = {
            e[0]: (getattr(e[1], "original_space", e[1]), e[2])
            for e in remote_spaces[0]
        }

        # Try to add the actual env's obs/action spaces.
        env_spaces = self.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: (env.observation_space, env.action_space)
            ),
            remote_worker_indices=[0],
            local_worker=False,
        )
        if env_spaces:
            # env_spaces group spaces by environment then worker.
            # So need to unpack thing twice.
            spaces["__env__"] = env_spaces[0][0]

        logger.info(
            "Inferred observation/action spaces from remote "
            f"worker (local worker has no env): {spaces}"
        )

        return spaces

    @DeveloperAPI
    def local_worker(self) -> RolloutWorker:
        """Returns the local rollout worker."""
        return self._local_worker

    @property
    def _remote_workers(self) -> List[ActorHandle]:
        """Returns the list of remote rollout workers."""
        if log_once("worker_set__remote_workers"):
            deprecation_warning(
                old="_remote_workers",
                help=(
                    "Accessing remote workers directly through "
                    "_remote_workers is strongly discouraged. "
                    "Please try to use one of the foreach accessors "
                    "that is fault tolerant. "
                ),
                error=False,
            )
        return list(self.__worker_manager.actors().values())

    def remote_workers(self) -> List[ActorHandle]:
        """Returns the list of remote rollout workers."""
        if log_once("worker_set_remote_workers"):
            deprecation_warning(
                old="remote_workers()",
                help=(
                    "Accessing the list of remote workers directly through "
                    "remote_workers() is strongly discouraged. "
                    "Please try to use one of the foreach accessors "
                    "that is fault tolerant. "
                ),
                error=False,
            )
        return list(self.__worker_manager.actors().values())

    @DeveloperAPI
    def num_remote_workers(self) -> int:
        return self.__worker_manager.num_actors()

    @DeveloperAPI
    def num_healthy_workers(self) -> int:
        if self._local_worker and self.__worker_manager.num_healthy_actors() <= 0:
            return 1
        # Return the number of healthy remote workers after this iteration.
        return self.__worker_manager.num_healthy_actors()

    @DeveloperAPI
    def sync_weights(
        self,
        policies: Optional[List[PolicyID]] = None,
        from_worker: Optional[RolloutWorker] = None,
        to_worker_indices: Optional[List[int]] = None,
        global_vars: Optional[Dict[str, TensorType]] = None,
    ) -> None:
        """Syncs model weights from the local worker to all remote workers.

        Args:
            policies: Optional list of PolicyIDs to sync weights for.
                If None (default), sync weights to/from all policies.
            from_worker: Optional RolloutWorker instance to sync from.
                If None (default), sync from this WorkerSet's local worker.
            global_vars: An optional global vars dict to set this
                worker to. If None, do not update the global_vars.
        """
        if self.local_worker() is None and from_worker is None:
            raise TypeError(
                "No `local_worker` in WorkerSet, must provide `from_worker` "
                "arg in `sync_weights()`!"
            )

        # Only sync if we have remote workers or `from_worker` is provided.
        weights = None
        if self.num_remote_workers() or from_worker is not None:
            weights = (from_worker or self.local_worker()).get_weights(policies)
            # Put weights only once into object store and use same object
            # ref to synch to all workers.
            weights_ref = ray.put(weights)

            def set_weight(w):
                w.set_weights(ray.get(weights_ref), global_vars)

            # Sync to all remote workers in this WorkerSet.
            # We take weight syncing as a good opportunity for health check,
            # because before a remote worker can re-join the lineup, we need
            # to bring them up to date anyways.
            # Any remote workers that have latest weights synced are healthy,
            # and will be able to participate in the next rounds of rollout or
            # evaluation.
            # If weight sync fails, the remote worker is taken out of service until:
            # 1. Ray Core potentially restarts the actor, and
            # 2. The worker gets updated during next round of weight syncing.
            # Do it on all remote workers regardless of their current state.
            self.foreach_worker(
                func=set_weight,
                healthy_only=False,
                remote_worker_indices=to_worker_indices,
            )

        # If `from_worker` is provided, also sync to this WorkerSet's
        # local worker.
        if self.local_worker() is not None:
            if from_worker is not None:
                self.local_worker().set_weights(weights, global_vars=global_vars)
            # If `global_vars` is provided and local worker exists  -> Update its
            # global_vars.
            elif global_vars is not None:
                self.local_worker().set_global_vars(global_vars)

    @DeveloperAPI
    def add_policy(
        self,
        policy_id: PolicyID,
        policy_cls: Optional[Type[Policy]] = None,
        policy: Optional[Policy] = None,
        *,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
        config: Optional[Union["AlgorithmConfig", PartialAlgorithmConfigDict]] = None,
        policy_state: Optional[PolicyState] = None,
        policy_mapping_fn: Optional[Callable[[AgentID, EpisodeID], PolicyID]] = None,
        policies_to_train: Optional[
            Union[
                Container[PolicyID],
                Callable[[PolicyID, Optional[SampleBatchType]], bool],
            ]
        ] = None,
        # Deprecated.
        workers: Optional[List[Union[RolloutWorker, ActorHandle]]] = DEPRECATED_VALUE,
    ) -> None:
        """Adds a policy to this WorkerSet's workers or a specific list of workers.

        Args:
            policy_id: ID of the policy to add.
            policy_cls: The Policy class to use for constructing the new Policy.
                Note: Only one of `policy_cls` or `policy` must be provided.
            policy: The Policy instance to add to this WorkerSet. If not None, the
                given Policy object will be directly inserted into the
                local worker and clones of that Policy will be created on all remote
                workers.
                Note: Only one of `policy_cls` or `policy` must be provided.
            observation_space: The observation space of the policy to add.
                If None, try to infer this space from the environment.
            action_space: The action space of the policy to add.
                If None, try to infer this space from the environment.
            config: The config object or overrides for the policy to add.
            policy_state: Optional state dict to apply to the new
                policy instance, right after its construction.
            policy_mapping_fn: An optional (updated) policy mapping function
                to use from here on. Note that already ongoing episodes will
                not change their mapping but will use the old mapping till
                the end of the episode.
            policies_to_train: An optional list of policy IDs to be trained
                or a callable taking PolicyID and SampleBatchType and
                returning a bool (trainable or not?).
                If None, will keep the existing setup in place. Policies,
                whose IDs are not in the list (or for which the callable
                returns False) will not be updated.
            workers: A list of RolloutWorker/ActorHandles (remote
                RolloutWorkers) to add this policy to. If defined, will only
                add the given policy to these workers.

        Raises:
            KeyError: If the given `policy_id` already exists in this WorkerSet.
        """
        if self.local_worker() and policy_id in self.local_worker().policy_map:
            raise KeyError(
                f"Policy ID '{policy_id}' already exists in policy map! "
                "Make sure you use a Policy ID that has not been taken yet."
                " Policy IDs that are already in your policy map: "
                f"{list(self.local_worker().policy_map.keys())}"
            )

        if workers is not DEPRECATED_VALUE:
            deprecation_warning(
                old="workers",
                help=(
                    "The `workers` argument to `WorkerSet.add_policy()` is deprecated "
                    "and a no-op now. Please do not use it anymore."
                ),
                error=False,
            )

        if (policy_cls is None) == (policy is None):
            raise ValueError(
                "Only one of `policy_cls` or `policy` must be provided to "
                "staticmethod: `WorkerSet.add_policy()`!"
            )
        validate_policy_id(policy_id, error=False)

        # Policy instance not provided: Use the information given here.
        if policy_cls is not None:
            new_policy_instance_kwargs = dict(
                policy_id=policy_id,
                policy_cls=policy_cls,
                observation_space=observation_space,
                action_space=action_space,
                config=config,
                policy_state=policy_state,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=list(policies_to_train)
                if policies_to_train
                else None,
            )
        # Policy instance provided: Create clones of this very policy on the different
        # workers (copy all its properties here for the calls to add_policy on the
        # remote workers).
        else:
            new_policy_instance_kwargs = dict(
                policy_id=policy_id,
                policy_cls=type(policy),
                observation_space=policy.observation_space,
                action_space=policy.action_space,
                config=policy.config,
                policy_state=policy.get_state(),
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=list(policies_to_train)
                if policies_to_train
                else None,
            )

        def _create_new_policy_fn(worker: RolloutWorker):
            # `foreach_worker` function: Adds the policy the the worker (and
            # maybe changes its policy_mapping_fn - if provided here).
            worker.add_policy(**new_policy_instance_kwargs)

        if self.local_worker() is not None:
            if policy is not None:
                self.local_worker().add_policy(
                    policy_id=policy_id,
                    policy=policy,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=policies_to_train,
                )
            else:
                self.local_worker().add_policy(**new_policy_instance_kwargs)

        # Add the policy to all remote workers.
        self.foreach_worker(_create_new_policy_fn, local_worker=False)

    @DeveloperAPI
    def add_workers(self, num_workers: int, validate: bool = False) -> None:
        """Creates and adds a number of remote workers to this worker set.

        Can be called several times on the same WorkerSet to add more
        RolloutWorkers to the set.

        Args:
            num_workers: The number of remote Workers to add to this
                WorkerSet.
            validate: Whether to validate remote workers after their construction
                process.

        Raises:
            RayError: If any of the constructed remote workers is not up and running
            properly.
        """
        old_num_workers = self.__worker_manager.num_actors()
        new_workers = [
            self._make_worker(
                cls=self._cls,
                env_creator=self._env_creator,
                validate_env=None,
                worker_index=old_num_workers + i + 1,
                num_workers=old_num_workers + num_workers,
                config=self._remote_config,
            )
            for i in range(num_workers)
        ]
        self.__worker_manager.add_actors(new_workers)

        # Validate here, whether all remote workers have been constructed properly
        # and are "up and running". Establish initial states.
        if validate:
            self.foreach_worker(lambda w: w.assert_healthy(), local_worker=False)

    @DeveloperAPI
    def reset(self, new_remote_workers: List[ActorHandle]) -> None:
        """Hard overrides the remote workers in this set with the given one.

        Args:
            new_remote_workers: A list of new RolloutWorkers
                (as `ActorHandles`) to use as remote workers.
        """
        self.__worker_manager.clear()
        self.__worker_manager.add_actors(new_remote_workers)

    # TODO(jungong) : to be deleted after we stop algorithms from
    # managing remote workers directly.
    def remove_failed_workers(self):
        faulty_indices = self._worker_health_check()
        removed_workers = []
        # Terminate faulty workers.
        for worker_index in faulty_indices:
            worker = self.__worker_manager.remove_actor(worker_index)
            logger.info(f"Trying to terminate faulty worker {worker_index}.")
            try:
                worker.__ray_terminate__.remote()
                removed_workers.append(worker)
            except Exception:
                logger.exception("Error terminating faulty worker.")

        if self.num_remote_workers() == 0:
            raise RuntimeError(
                f"No healthy workers remaining (worker indices {faulty_indices} have "
                f"died)! Can't continue training."
            )
        return removed_workers

    # TODO(jungong) : to be deleted after we stop algorithms from
    # managing remote workers directly.
    def recreate_failed_workers(
        self, local_worker_for_synching: RolloutWorker
    ) -> Tuple[List[ActorHandle], List[ActorHandle]]:
        """Recreates any failed workers (after health check).

        Args:
            local_worker_for_synching: RolloutWorker to use to synchronize the weights
                after recreation.

        Returns:
            A tuple consisting of two items: The list of removed workers and the list of
            newly added ones.
        """
        faulty_indices = self._worker_health_check()
        removed_workers = []
        new_workers = []
        for worker_index in faulty_indices:
            worker = self.__worker_manager.remove_actor(worker_index)
            removed_workers.append(worker)
            logger.info(f"Trying to recreate faulty worker {worker_index}")
            try:
                worker.__ray_terminate__.remote()
            except Exception:
                logger.exception("Error terminating faulty worker.")

            # Try to recreate the failed worker (start a new one).
            new_worker = self._make_worker(
                cls=self._cls,
                env_creator=self._env_creator,
                validate_env=None,
                worker_index=worker_index + 1,
                num_workers=len(self._remote_workers),
                recreated_worker=True,
                config=self._remote_config,
            )

            # Sync new worker from provided one (or local one).
            # Restore weights and global variables.
            new_worker.set_state.remote(
                state=local_worker_for_synching.get_state(),
            )

            # Add new worker to list of remote workers.
            self.__worker_manager.add_actors([new_worker])
            new_workers.append(new_worker)

        return removed_workers, new_workers

    @DeveloperAPI
    def stop(self) -> None:
        """Calls `stop` on all rollout workers (including the local one)."""
        try:
            # Make sure we stop all workers, include the ones that were just
            # restarted / recovered.
            self.foreach_worker(
                lambda w: w.stop(), healthy_only=False, local_worker=True
            )
        except Exception:
            logger.exception("Failed to stop workers!")
        finally:
            self.__worker_manager.clear()

    @DeveloperAPI
    def is_policy_to_train(
        self, policy_id: PolicyID, batch: Optional[SampleBatchType] = None
    ) -> bool:
        """Whether given PolicyID (optionally inside some batch) is trainable."""
        local_worker = self.local_worker()
        if local_worker:
            if local_worker.is_policy_to_train is None:
                return True
            return local_worker.is_policy_to_train(policy_id, batch)
        else:
            raise NotImplementedError

    @DeveloperAPI
    def foreach_worker(
        self,
        func: Callable[[RolloutWorker], T],
        *,
        local_worker=True,
        # TODO(jungong) : switch to True once Algorithm is migrated.
        healthy_only=False,
        remote_worker_indices: List[int] = None,
        timeout_seconds=None,
    ) -> List[T]:
        """Calls the given function with each worker instance as the argument.

        Args:
            func: The function to call for each worker (as only arg).
            local_worker: Whether apply func on local worker too. Default is True.
            healthy_only: Apply func on known active workers only. By default
                this will apply func on all workers regardless of their states.
            remote_worker_indices: Apply func on a selected set of remote workers.
            timeout_seconds: Time to wait for results. Default is None.

        Returns:
             The list of return values of all calls to `func([worker])`.
        """
        local_result = []
        if local_worker and self.local_worker() is not None:
            local_result = [func(self.local_worker())]

        remote_results = self.__worker_manager.foreach_actor(
            func,
            healthy_only=healthy_only,
            remote_actor_ids=remote_worker_indices,
            timeout_seconds=timeout_seconds,
        )
        remote_results = [r.get() for r in remote_results.ignore_ray_errors()]

        return local_result + remote_results

    @DeveloperAPI
    def foreach_worker_async(
        self,
        func: Callable[[RolloutWorker], T],
        *,
        # TODO(jungong) : switch to True once Algorithm is migrated.
        healthy_only=False,
        remote_worker_indices: List[int] = None,
    ) -> int:
        """Calls the given function with each worker instance as the argument.

        Args:
            func: The function to call for each worker (as only arg).
            healthy_only: Apply func on known active workers only. By default
                this will apply func on all workers regardless of their states.
            remote_worker_indices: Apply func on a selected set of remote workers.

        Returns:
             The number of async requests that are actually fired.
        """
        return self.__worker_manager.foreach_actor_async(
            func,
            healthy_only=healthy_only,
            remote_actor_ids=remote_worker_indices,
        )

    @DeveloperAPI
    def fetch_ready_async_reqs(
        self,
        *,
        timeout_seconds=0,
    ) -> Tuple[List[int], List[T]]:
        """Get results from outstanding asynchronous requests that are ready.

        Args:
            timeout_seconds: Time to wait for results. Default is 0, meaning
                those requests that are already ready.

        Returns:
            A list of results successfully returned from outstanding remote calls,
            paired with the indices of the callee workers.
        """
        remote_result = self.__worker_manager.fetch_ready_async_reqs(
            timeout_seconds=timeout_seconds
        )
        return [(r.actor_id, r.get()) for r in remote_result.ignore_ray_errors()]

    @DeveloperAPI
    def foreach_policy(self, func: Callable[[Policy, PolicyID], T]) -> List[T]:
        """Calls `func` with each worker's (policy, PolicyID) tuple.

        Note that in the multi-agent case, each worker may have more than one
        policy.

        Args:
            func: A function - taking a Policy and its ID - that is
                called on all workers' Policies.

        Returns:
            The list of return values of func over all workers' policies. The
                length of this list is:
                (num_workers + 1 (local-worker)) *
                [num policies in the multi-agent config dict].
                The local workers' results are first, followed by all remote
                workers' results
        """
        results = []
        for r in self.foreach_worker(
            lambda w: w.foreach_policy(func), local_worker=True
        ):
            results.extend(r)
        return results

    @DeveloperAPI
    def foreach_policy_to_train(self, func: Callable[[Policy, PolicyID], T]) -> List[T]:
        """Apply `func` to all workers' Policies iff in `policies_to_train`.

        Args:
            func: A function - taking a Policy and its ID - that is
                called on all workers' Policies, for which
                `worker.is_policy_to_train()` returns True.

        Returns:
            List[any]: The list of n return values of all
                `func([trainable policy], [ID])`-calls.
        """
        results = []
        for r in self.foreach_worker(
            lambda w: w.foreach_policy_to_train(func), local_worker=True
        ):
            results.extend(r)
        return results

    @DeveloperAPI
    def foreach_env(self, func: Callable[[EnvType], List[T]]) -> List[List[T]]:
        """Calls `func` with all workers' sub-environments as args.

        An "underlying sub environment" is a single clone of an env within
        a vectorized environment.
        `func` takes a single underlying sub environment as arg, e.g. a
        gym.Env object.

        Args:
            func: A function - taking an EnvType (normally a gym.Env object)
                as arg and returning a list of lists of return values, one
                value per underlying sub-environment per each worker.

        Returns:
            The list (workers) of lists (sub environments) of results.
        """
        return list(
            self.foreach_worker(
                lambda w: w.foreach_env(func),
                local_worker=True,
            )
        )

    @DeveloperAPI
    def foreach_env_with_context(
        self, func: Callable[[BaseEnv, EnvContext], List[T]]
    ) -> List[List[T]]:
        """Calls `func` with all workers' sub-environments and env_ctx as args.

        An "underlying sub environment" is a single clone of an env within
        a vectorized environment.
        `func` takes a single underlying sub environment and the env_context
        as args.

        Args:
            func: A function - taking a BaseEnv object and an EnvContext as
                arg - and returning a list of lists of return values over envs
                of the worker.

        Returns:
            The list (1 item per workers) of lists (1 item per sub-environment)
                of results.
        """
        return list(
            self.foreach_worker(
                lambda w: w.foreach_env_with_context(func),
                local_worker=True,
            )
        )

    @staticmethod
    def _from_existing(
        local_worker: RolloutWorker, remote_workers: List[ActorHandle] = None
    ):
        workers = WorkerSet(
            env_creator=None, default_policy_class=None, config=None, _setup=False
        )
        workers._local_worker = local_worker
        workers.__worker_manager.add_actors(remote_workers or [])
        return workers

    def _make_worker(
        self,
        *,
        cls: Callable,
        env_creator: EnvCreator,
        validate_env: Optional[Callable[[EnvType], None]],
        worker_index: int,
        num_workers: int,
        recreated_worker: bool = False,
        config: "AlgorithmConfig",
        spaces: Optional[
            Dict[PolicyID, Tuple[gym.spaces.Space, gym.spaces.Space]]
        ] = None,
    ) -> Union[RolloutWorker, ActorHandle]:
        def session_creator():
            logger.debug("Creating TF session {}".format(config["tf_session_args"]))
            return tf1.Session(config=tf1.ConfigProto(**config["tf_session_args"]))

        worker = cls(
            env_creator=env_creator,
            validate_env=validate_env,
            default_policy_class=self._policy_class,
            tf_session_creator=(session_creator if config["tf_session_args"] else None),
            config=config,
            worker_index=worker_index,
            num_workers=num_workers,
            recreated_worker=recreated_worker,
            log_dir=self._logdir,
            spaces=spaces,
            dataset_shards=self._ds_shards,
        )

        return worker

    # TODO(jungong) : to be deleted after we stop algorithms from
    # managing remote workers directly.
    def _worker_health_check(self) -> List[int]:
        """Performs a health-check on each remote worker.

        Returns:
            List of indices (into `self._remote_workers` list) of faulty workers.
            Note that index=1 is the 0th item in `self._remote_workers`.
        """
        logger.info("Health checking all workers ...")

        remote_results = self.__worker_manager.foreach_actor(
            lambda w: w.sample_with_count(),
            healthy_only=False,
        )

        return [
            r.actor_id
            for r in remote_results
            if not r.ok and isinstance(r.get(), RayError)
        ]

    @classmethod
    def _valid_module(cls, class_path):
        del cls
        if (
            isinstance(class_path, str)
            and not os.path.isfile(class_path)
            and "." in class_path
        ):
            module_path, class_name = class_path.rsplit(".", 1)
            try:
                spec = importlib.util.find_spec(module_path)
                if spec is not None:
                    return True
            except (ModuleNotFoundError, ValueError):
                print(
                    f"module {module_path} not found while trying to get "
                    f"input {class_path}"
                )
        return False

    @Deprecated(new="WorkerSet.foreach_policy_to_train", error=True)
    def foreach_trainable_policy(self, func):
        return self.foreach_policy_to_train(func)

    @Deprecated(new="WorkerSet.is_policy_to_train([pid], [batch]?)", error=True)
    def trainable_policies(self):
        pass
