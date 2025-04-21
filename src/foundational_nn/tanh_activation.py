import numpy as np

from .activation_function import ActivationFunction

class TanhActivation(ActivationFunction):
    """Tanh activation function."""

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)