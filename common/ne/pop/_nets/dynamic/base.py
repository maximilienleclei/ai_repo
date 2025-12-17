import torch
from common.ne.pop._nets.base import BaseNets, BaseNetsConfig

class DynamicNets(BaseNets):

    def __init__(self, config: BaseNetsConfig):
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def mutate(self) -> None:
        ...
