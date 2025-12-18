from hydra_zen import ZenStore

from common.ne.runner import NeuroevolutionTaskRunner


class TaskRunner(NeuroevolutionTaskRunner):
    """Task runner for discrete control neuroevolution experiments.

    Supports feedforward, recurrent, and dynamic network architectures.
    """

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store)


TaskRunner.run_task()
