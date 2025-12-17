import torch
from common.ne.pop._nets.static.base import BaseStaticNets, StaticNetsConfig

class RecurrentStaticNets(BaseStaticNets):

    def __init__(self, config: StaticNetsConfig):
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def mutate(self) -> None:
        ...
