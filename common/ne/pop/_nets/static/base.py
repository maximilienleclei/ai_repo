from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from common.ne.pop._nets.base import BaseNets, BaseNetsConfig

@dataclass
class StaticNetsConfig(BaseNetsConfig):
    layer_dims: list[int]
    sigma: float = 1e-3
    sigma_sigma: float | None = 1e-2

class BaseStaticNets(BaseNets, ABC):

    def __init__(self, config: StaticNetsConfig):
        ...

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def mutate(self) -> None:
        ...
