import torch
from common.ne.pop.base import BasePop
from common.ne.alg.base import BaseAlg

class SimpleES(BaseAlg):

    def __call__(self, population: BasePop, fitness_scores: torch.Tensor) -> None:
        ...
