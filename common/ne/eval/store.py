from hydra_zen import ZenStore
from common.ne.eval._optim.store import store_configs as store_optim_eval_configs


def store_configs(store: ZenStore) -> None:
    store_optim_eval_configs(store)
