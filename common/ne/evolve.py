from functools import partial
import torch
from common.ne.config import NeuroevolutionSubtaskConfig
from common.ne.alg.base import BaseAlg
from common.ne.eval.base import BaseEval
from common.ne.pop.base import BasePop

def evolve(
    alg: BaseAlg,
    eval: BaseEval,
    pop: BasePop,
    config: NeuroevolutionSubtaskConfig,
) -> float:
    ...
