import numpy as np

from .activation_function import ActivationFunction

class ReLUActivation(ActivationFunction):
    """ReLU activation function."""

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)