import numpy as np
from abc import ABCMeta, abstractmethod
from . import Frame

## Provides occlusion mask (interface for accurate AI code generation)
class OcclusionProvider(metaclass=ABCMeta):
    @abstractmethod
    def occlusion(self, frame: Frame) -> np.ndarray:
        pass
