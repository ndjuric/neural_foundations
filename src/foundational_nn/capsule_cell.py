import numpy as np
from typing import Callable

from .cell import Cell

class CapsuleCell(Cell):
    """A cell type that represents entities as vectors ("capsules")."""

    def __init__(self, inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray, activation: Callable[[np.ndarray], np.ndarray], routing_iterations: int) -> None:
        super().__init__(inputs=inputs)
        self.weights: np.ndarray = weights
        self.bias: np.ndarray = bias
        self.activation: Callable[[np.ndarray], np.ndarray] = activation
        self.routing_iterations: int = routing_iterations

    def compute(self) -> np.ndarray:
        weighted_sum = np.dot(self.inputs, self.weights) + self.bias
        self.outputs = self.activation(weighted_sum)
        return self.outputs

    def visualize(self) -> str:
        return f"CapsuleCell[{self.outputs}]"