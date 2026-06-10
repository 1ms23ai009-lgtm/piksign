from torch import nn, Tensor
from abc import abstractmethod
from typing import Any, Callable


class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


