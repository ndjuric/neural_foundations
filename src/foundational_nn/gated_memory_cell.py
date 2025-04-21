import numpy as np
from typing import Callable

from .cell import Cell

class GatedMemoryCell(Cell):
    """Gated memory cell (generalization of LSTM's memory cell)."""

    def __init__(self, gates_output: np.ndarray, cell_state_t_1: np.ndarray) -> None:
        super().__init__(inputs=None)
        self.gates_output: np.ndarray = gates_output
        self.cell_state_t_1: np.ndarray = cell_state_t_1

    def compute(self, gate_logic: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        self.outputs = gate_logic(self.gates_output, self.cell_state_t_1)
        return self.outputs

    def visualize(self) -> str:
        return f"GatedMemoryCell[{self.outputs}]"