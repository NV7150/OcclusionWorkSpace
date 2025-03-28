import numpy as np
from abc import ABCMeta, abstractmethod

## Provides occlusion mask (interface for accurate AI code generation)
class OcclusionProvider(metaclass=ABCMeta):
    @abstractmethod
    def occlusion(self) -> np.ndarray:
        pass
