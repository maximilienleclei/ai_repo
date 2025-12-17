from hydra_zen import ZenStore
from common.utils.hydra_zen import generate_config
from common.ne.eval._optim.supervised import SupervisedEval
from common.ne.eval._optim.score import ScoreEval
from common.ne.eval._optim.imitate import ImitateEval


def store_configs(store: ZenStore) -> None:
    store(generate_config(SupervisedEval), name="supervised", group="eval/optim_eval")
    store(generate_config(ScoreEval), name="score", group="eval/optim_eval")
    store(generate_config(ImitateEval), name="imitate", group="eval/optim_eval")
