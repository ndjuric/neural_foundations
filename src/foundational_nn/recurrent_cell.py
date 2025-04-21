import numpy as np
from typing import Callable

from .cell import Cell

class RecurrentCell(Cell):
    """Processes sequential data with a feedback loop."""

    def __init__(self, input_t: np.ndarray, hidden_state_t_1: np.ndarray, weight_in: np.ndarray, weight_hidden: np.ndarray, bias: float, activation: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(inputs=None)
        self.input_t: np.ndarray = input_t
        self.hidden_state_t_1: np.ndarray = hidden_state_t_1
        self.weight_in: np.ndarray = weight_in
        self.weight_hidden: np.ndarray = weight_hidden
        self.bias: float = bias
        self.activation: Callable[[np.ndarray], np.ndarray] = activation

    def compute(self) -> np.ndarray:
        new_hidden_state = self.activation(
            np.dot(self.input_t, self.weight_in)
            + np.dot(self.hidden_state_t_1, self.weight_hidden)
            + self.bias
        )
        self.outputs = new_hidden_state
        return self.outputs

    def visualize(self) -> str:
        return f"RecurrentCell[{self.outputs}]"