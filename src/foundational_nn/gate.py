import numpy as np
from typing import Callable

from .cell import Cell

class Gate(Cell):
    """Gating mechanism (LSTM-like)."""

    def __init__(self, inputs: np.ndarray, weights: np.ndarray, bias: float, activation: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(inputs=inputs)
        self.weights: np.ndarray = weights
        self.bias: float = bias
        self.activation: Callable[[np.ndarray], np.ndarray] = activation

    def compute(self) -> np.ndarray:
        self.outputs = self.activation(np.dot(self.inputs, self.weights) + self.bias)
        return self.outputs

    def visualize(self) -> str:
        return f"Gate[{self.outputs}]"