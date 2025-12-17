from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class BaseNetsConfig:
    num_nets: int

class BaseNets(ABC):

    def __init__(self, config: BaseNetsConfig):
        ...

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def mutate(self) -> None:
        ...
