from common.ne.eval._optim.base import BaseOptimEval
from common.ne.eval._test.base import BaseTestEval

class BaseEval:

    def __init__(self, optim_eval: BaseOptimEval, test_evals: list[BaseTestEval]):
        self.optim_eval = optim_eval
        self.test_evals = test_evals
