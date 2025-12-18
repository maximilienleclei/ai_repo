from hydra_zen import ZenStore

from common.ne.pop._nets.base import BaseNets
from common.ne.pop._nets.store import store_configs as store_nets_configs
from common.ne.pop.actor import ActorPop
from common.ne.pop.base import BasePopConfig
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store_nets_configs(store)
    store(
        generate_config(ActorPop, config=generate_config(BasePopConfig)),
        name="actor",
        group="pop_",
    )
