from hydra_zen import ZenStore

from common.ne.pop._nets.base import BaseNetsConfig
from common.ne.pop._nets.dynamic.base import DynamicNets
from common.ne.pop._nets.static.base import StaticNetsConfig
from common.ne.pop._nets.static.feedforward import FeedforwardStaticNets
from common.ne.pop._nets.static.recurrent import RecurrentStaticNets
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store = store(group="pop_/nets")
    store(
        generate_config(
            FeedforwardStaticNets, config=generate_config(StaticNetsConfig)
        ),
        name="feedforward",
    )
    store(
        generate_config(
            RecurrentStaticNets, config=generate_config(StaticNetsConfig)
        ),
        name="recurrent",
    )
    store(
        generate_config(DynamicNets, config=generate_config(BaseNetsConfig)),
        name="dynamic",
    )
