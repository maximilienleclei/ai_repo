from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from common.ne.pop.nets.base import BaseNets, BaseNetsConfig

@dataclass
class StaticNetsConfig(BaseNetsConfig):
    layer_dims: list[int]

class BaseStaticNets(BaseNets, ABC):

    def __init__(self, config: StaticNetsConfig):
        ...

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def mutate(self) -> None:
        ...
