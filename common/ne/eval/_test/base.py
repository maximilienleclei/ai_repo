import torch

from common.ne.pop.base import BasePop


class BaseTestEval:

    def __call__(self, population: BasePop) -> float: ...
