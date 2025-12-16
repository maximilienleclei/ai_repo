from hydra_zen import ZenStore
from lightning.pytorch import Trainer
from utils.hydra_zen import generate_config_partial

from common.dl.config import DeepLearningTaskConfig
from common.dl.litmodule.store import store_configs as store_litmodule_configs


def store_configs(store: ZenStore) -> None:
    store(DeepLearningTaskConfig, name="config")
    store_litmodule_configs(store)
    store(
        generate_config_partial(
            Trainer,
            accelerator="${config.device}",
            default_root_dir="${config.output_dir}/lightning/",
            gradient_clip_val=1.0,
        ),
        name="base",
        group="trainer",
    )
