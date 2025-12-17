from abc import ABC, abstractmethod
import torch
from common.ne.pop.base import BasePop

class BaseAlg(ABC):

    @abstractmethod
    def __call__(self, population: BasePop, fitness_scores: torch.Tensor) -> None:
        ...
