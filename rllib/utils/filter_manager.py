import logging
from typing import Optional

import ray
from ray.rllib.utils.annotations import DeveloperAPI
from ray.exceptions import GetTimeoutError

logger = logging.getLogger(__name__)


@DeveloperAPI
class FilterManager:
    """Manages filters and coordination across remote evaluators that expose
    `get_filters` and `sync_filters`.
    """

    @staticmethod
    @DeveloperAPI
    def synchronize(
        local_filters,
        worker_set,
        update_remote=True,
        timeout_seconds: Optional[float] = None,
    ):
        """Aggregates all filters from remote evaluators.

        Local copy is updated and then broadcasted to all remote evaluators.

        Args:
            local_filters: Filters to be synchronized.
            worker_set: Remote WorkerSet with filters.
            update_remote: Whether to push updates to remote filters.
            timeout_seconds: How long to wait for filter to get or set filters
        """
        logger.info("Synchronizing filters ...")

        remote_filters = worker_set.foreach_worker(
            func=lambda worker: worker.get_filters(flush_after=True),
            timeout_seconds=timeout_seconds,
        )

        for rf in remote_filters:
            for k in local_filters:
                local_filters[k].apply_changes(rf[k], with_buffer=False)

        if update_remote:
            copies = {k: v.as_serializable() for k, v in local_filters.items()}
            remote_copy = ray.put(copies)

            logger.info("Updating remote filters ...")
            worker_set.foreach_worker(
                func=lambda worker: worker.sync_filters(remote_copy),
                timeout_seconds=timeout_seconds,
            )
