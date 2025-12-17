from dataclasses import dataclass, field
from typing import Any

from hydra_zen import make_config

from common.config import BaseSubtaskConfig
from common.ne.alg.base import BaseAlg
from common.ne.eval.base import BaseEval
from common.ne.pop.base import BasePop
from common.utils.hydra_zen import generate_config


@dataclass
class NeuroevolutionSubtaskConfig(BaseSubtaskConfig):
    num_minutes: int = 60


@dataclass
class NeuroevolutionTaskConfig(
    make_config(
        alg=generate_config(BaseAlg),
        eval=generate_config(BaseEval),
        pop=generate_config(BasePop),
        config=generate_config(NeuroevolutionSubtaskConfig),
    ),
):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            "project",
            "task",
            {"task": None},
        ],
    )
