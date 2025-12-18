import torch

from common.ne.alg.base import BaseAlg
from common.ne.pop.base import BasePop


class SimpleGA(BaseAlg):

    def __call__(
        self, population: BasePop, fitness_scores: torch.Tensor
    ) -> None:
        sorted_indices = torch.argsort(fitness_scores, descending=True)
        half = population.nets.config.num_nets // 2
        top_half_indices = sorted_indices[:half]
        indices = torch.cat([top_half_indices, top_half_indices])
        population.nets.resample(indices)
