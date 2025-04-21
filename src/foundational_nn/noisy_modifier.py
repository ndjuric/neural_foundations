import numpy as np

from .cell import Cell
from .modifiable import Modifiable

class NoisyModifier(Modifiable):
    """Adds noise to cell output."""

    def apply_modifier(self, cell: Cell, noise_std: float = 1.0) -> None:
        if not isinstance(cell, Cell):
            raise TypeError("Cannot apply NoisyModifier to this type")
        original_compute = cell.compute

        def noisy_compute() -> np.ndarray:
            result = original_compute()
            return result + np.random.normal(0, noise_std, size=result.shape)

        cell.compute = noisy_compute