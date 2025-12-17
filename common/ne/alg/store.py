from hydra_zen import ZenStore
from common.utils.hydra_zen import generate_config
from common.ne.alg.simple_ga import SimpleGA
from common.ne.alg.simple_es import SimpleES
from common.ne.alg.cma_es import CMAES


def store_configs(store: ZenStore) -> None:
    store(generate_config(SimpleGA), name="simple_ga", group="alg")
    store(generate_config(SimpleES), name="simple_es", group="alg")
    store(generate_config(CMAES), name="cma_es", group="alg")
