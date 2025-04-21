import numpy as np

from .activation_function import ActivationFunction

class SigmoidActivation(ActivationFunction):
    """Sigmoid activation function."""

    def compute(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))