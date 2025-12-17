from abc import ABC, abstractmethod

import torch

from common.ne.pop.base import BasePop


class BaseOptimEval(ABC):

    @abstractmethod
    def __call__(self, population: BasePop) -> torch.Tensor: ...
