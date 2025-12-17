from hydra_zen import ZenStore
from common.ne.pop._nets.store import store_configs as store_nets_configs


def store_configs(store: ZenStore) -> None:
    store_nets_configs(store)
