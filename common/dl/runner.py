from functools import partial
from typing import Any

from hydra_zen import ZenStore
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from common.dl.config import DeepLearningSubtaskConfig
from common.dl.datamodule.base import BaseDataModule
from common.dl.litmodule.base import BaseLitModule
from common.dl.store import store_configs as store_dl_configs
from common.dl.train import train
from common.runner import BaseTaskRunner


class DeepLearningTaskRunner(BaseTaskRunner):
    @classmethod
    def store_configs(
        cls: type["DeepLearningTaskRunner"],
        store: ZenStore,
    ) -> None:
        super().store_configs(store)
        store_dl_configs(store)

    @classmethod
    def run_subtask(
        cls: type["DeepLearningTaskRunner"],
        trainer: partial[Trainer],
        datamodule: BaseDataModule,
        litmodule: BaseLitModule,
        logger: partial[WandbLogger],
        config: DeepLearningSubtaskConfig,
    ) -> Any:
        return train(trainer, datamodule, litmodule, logger, config)
