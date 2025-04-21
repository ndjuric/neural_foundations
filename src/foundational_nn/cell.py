from abc import abstractmethod
from typing import Optional

import numpy as np

from .visualizable import Visualizable

class Cell(Visualizable):
    """Abstract base class for a basic cell."""

    def __init__(self, inputs: Optional[np.ndarray] = None, outputs: Optional[np.ndarray] = None):
        self.inputs: Optional[np.ndarray] = inputs
        self.outputs: Optional[np.ndarray] = outputs

    @abstractmethod
    def compute(self) -> np.ndarray:
        """Compute the cell's output."""
        pass