from hydra_zen import ZenStore

from common.ne.eval.score import ScoreEval, ScoreEvalConfig
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store(
        generate_config(ScoreEval, config=generate_config(ScoreEvalConfig)),
        name="score",
        group="eval",
    )
