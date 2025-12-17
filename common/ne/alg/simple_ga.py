import torch

from common.ne.alg.base import BaseAlg
from common.ne.pop.base import BasePop


class SimpleGA(BaseAlg):

    def __call__(
        self, population: BasePop, fitness_scores: torch.Tensor
    ) -> None: ...
