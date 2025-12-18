from typing import Any

from hydra_zen import ZenStore

from common.ne.alg.base import BaseAlg
from common.ne.config import NeuroevolutionSubtaskConfig
from common.ne.eval.base import BaseEval
from common.ne.evolve import evolve
from common.ne.pop.base import BasePop
from common.ne.store import store_configs as store_ne_configs
from common.runner import BaseTaskRunner


class NeuroevolutionTaskRunner(BaseTaskRunner):
    @classmethod
    def store_configs(
        cls: type["NeuroevolutionTaskRunner"],
        store: ZenStore,
    ) -> None:
        super().store_configs(store)
        store_ne_configs(store)

    @classmethod
    def run_subtask(
        cls: type["NeuroevolutionTaskRunner"],
        alg: BaseAlg,
        eval: BaseEval,
        pop_: BasePop,
        config: NeuroevolutionSubtaskConfig,
    ) -> Any:
        return evolve(alg, eval, pop_, config)
