import numpy as np

from .cell import Cell

class InputCell(Cell):
    """Represents a single input value."""

    def __init__(self, value: float) -> None:
        super().__init__(inputs=np.array([value]))
        self.value: float = value

    def compute(self) -> np.ndarray:
        self.outputs = np.array([self.value])
        return self.outputs

    def visualize(self) -> str:
        return f"InputCell[{self.outputs}]"