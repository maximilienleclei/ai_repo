from abc import ABC
from common.ne.pop._nets.base import BaseNets

class BasePop(ABC):

    def __init__(self, nets: BaseNets):
        ...
