from hydra_zen import ZenStore

from common.ne.config import NeuroevolutionTaskConfig
from common.ne.alg.store import store_configs as store_alg_configs
from common.ne.eval.store import store_configs as store_eval_configs
from common.ne.pop.store import store_configs as store_pop_configs


def store_configs(store: ZenStore) -> None:
    store(NeuroevolutionTaskConfig, name="config")
    store_alg_configs(store)
    store_eval_configs(store)
    store_pop_configs(store)
