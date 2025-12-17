from hydra_zen import ZenStore
from common.utils.hydra_zen import generate_config
from common.ne.pop._nets.static.feedforward import FeedforwardStaticNets
from common.ne.pop._nets.static.recurrent import RecurrentStaticNets
from common.ne.pop._nets.static.base import StaticNetsConfig


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
