import numpy as np

from .cell import Cell
from .modifiable import Modifiable

class ProbabilisticModifier(Modifiable):
    """Makes cell output probabilistic."""

    def apply_modifier(self, cell: Cell, mean: float = 0.0, std_dev: float = 1.0) -> None:
        if not isinstance(cell, Cell):
            raise TypeError("Cannot apply ProbabilisticModifier to this type")
        original_compute = cell.compute

        def probabilistic_compute() -> np.ndarray:
            _ = original_compute()
            return np.random.normal(mean, std_dev, size=cell.outputs.shape)

        cell.compute = probabilistic_compute