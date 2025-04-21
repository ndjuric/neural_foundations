import numpy as np

from .cell import Cell

class MemoryCell(Cell):
    """Core memory unit for LSTM-like cells."""

    def __init__(self, forget_gate_output: np.ndarray, input_gate_output: np.ndarray, cell_gate_output: np.ndarray, previous_cell_state: np.ndarray) -> None:
        super().__init__(inputs=None)
        self.forget_gate_output: np.ndarray = forget_gate_output
        self.input_gate_output: np.ndarray = input_gate_output
        self.cell_gate_output: np.ndarray = cell_gate_output
        self.previous_cell_state: np.ndarray = previous_cell_state

    def compute(self) -> np.ndarray:
        new_cell_state = self.forget_gate_output * self.previous_cell_state + self.input_gate_output * self.cell_gate_output
        self.outputs = new_cell_state
        return self.outputs

    def visualize(self) -> str:
        return f"MemoryCell[{self.outputs}]"