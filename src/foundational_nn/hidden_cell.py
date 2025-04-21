import numpy as np
from typing import Callable

from .cell import Cell

class HiddenCell(Cell):
    """Intermediate cell for processing data."""

    def __init__(self, inputs: np.ndarray, weights: np.ndarray, bias: float, activation: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(inputs=inputs)
        self.weights: np.ndarray = weights
        self.bias: float = bias
        self.activation: Callable[[np.ndarray], np.ndarray] = activation

    def compute(self) -> np.ndarray:
        weighted_sum = np.dot(self.inputs, self.weights) + self.bias
        self.outputs = self.activation(weighted_sum)
        return self.outputs

    def visualize(self) -> str:
        return f"HiddenCell[{self.outputs}]"