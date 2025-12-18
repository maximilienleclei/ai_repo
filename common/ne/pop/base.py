from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from common.ne.pop._nets.base import BaseNets


@dataclass
class BasePopConfig:
    size: int


class BasePop(ABC):
    def __init__(self, config: BasePopConfig, nets: BaseNets):
        self.config = config
        self.nets = nets

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
