from abc import ABC, abstractmethod

import torch

from common.ne.eval._optim.base import BaseOptimEval
from common.ne.pop.base import BasePop


class ScoreEval(BaseOptimEval):

    def __call__(self, population: BasePop) -> torch.Tensor: ...
