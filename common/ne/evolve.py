import time

from common.ne.alg.base import BaseAlg
from common.ne.config import NeuroevolutionSubtaskConfig
from common.ne.eval.base import BaseEval
from common.ne.pop.base import BasePop


def evolve(
    alg: BaseAlg,
    eval: BaseEval,
    pop_: BasePop,
    config: NeuroevolutionSubtaskConfig,
) -> float:
    start_time = time.time()
    generation = 0
    while (time.time() - start_time) / 60 < config.num_minutes:
        pop_.nets.mutate()
        fitness_scores = eval(pop_)
        alg(pop_, fitness_scores)
        print(
            f"Gen {generation}: best = {fitness_scores.max():.2f}, mean = {fitness_scores.mean():.2f}"
        )
        generation += 1

    return None
