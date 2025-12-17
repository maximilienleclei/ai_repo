import torch
from common.ne.alg.base import BaseAlg
from common.ne.eval.base import BaseEval
from common.ne.pop.base import BasePop

def evolve(alg: BaseAlg, eval: BaseEval, pop: BasePop):
    ...
