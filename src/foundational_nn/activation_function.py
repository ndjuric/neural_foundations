from abc import ABC, abstractmethod

import numpy as np

class ActivationFunction(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        """Compute the activation function."""
        pass