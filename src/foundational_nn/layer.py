from abc import abstractmethod
from typing import Optional

import numpy as np

from .visualizable import Visualizable

class Layer(Visualizable):
    """Abstract base class for a layer."""

    def __init__(self, inputs: Optional[np.ndarray] = None, outputs: Optional[np.ndarray] = None):
        self.inputs: Optional[np.ndarray] = inputs
        self.outputs: Optional[np.ndarray] = outputs

    @abstractmethod
    def compute(self, *args, **kwargs) -> np.ndarray:
        """Compute the layer's output."""
        pass