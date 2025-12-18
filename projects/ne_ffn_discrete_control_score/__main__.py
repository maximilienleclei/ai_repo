from hydra_zen import ZenStore

from common.ne.runner import NeuroevolutionTaskRunner


class TaskRunner(NeuroevolutionTaskRunner):
    """Task runner for feedforward control experiments."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store)


TaskRunner.run_task()
