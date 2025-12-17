from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class BaseNetsConfig:
    num_nets: int
    num_inputs: int
    num_outputs: int

class BaseNets(ABC):

    def __init__(self, config: BaseNetsConfig):
        ...

    @abstractmethod
    def __call__(self, x: torch.Tensor, mem: Any = None) -> tuple[torch.Tensor, Any]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor
            mem: Memory state (None for stateless nets, or net-specific state)

        Returns:
            (output, new_mem): Output tensor and updated memory state
        """
        ...

    def mutate(self) -> None:
        ...
