from hydra_zen import ZenStore
from common.utils.hydra_zen import generate_config
from common.ne.pop._nets.base import BaseNetsConfig
from common.ne.pop._nets.static.feedforward import FeedforwardStaticNets
from common.ne.pop._nets.static.recurrent import RecurrentStaticNets
from common.ne.pop._nets.static.base import StaticNetsConfig
from common.ne.pop._nets.dynamic.base import DynamicNets


def store_configs(store: ZenStore) -> None:
    store(
        generate_config(FeedforwardStaticNets, config=generate_config(StaticNetsConfig)),
        name="feedforward",
        group="pop/nets",
    )
    store(
        generate_config(RecurrentStaticNets, config=generate_config(StaticNetsConfig)),
        name="recurrent",
        group="pop/nets",
    )
    store(
        generate_config(DynamicNets, config=generate_config(BaseNetsConfig)),
        name="dynamic",
        group="pop/nets",
    )
