import torch
from common.ne.pop.base import BasePop
from common.ne.eval._test.base import BaseTestEval

class EnvEval(BaseTestEval):

    def __call__(self, population: BasePop) -> float:
        ...
