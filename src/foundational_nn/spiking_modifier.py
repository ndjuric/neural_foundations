import numpy as np

from .cell import Cell
from .modifiable import Modifiable

class SpikingModifier(Modifiable):
    """Makes cell behave like a spiking neuron."""

    def apply_modifier(self, cell: Cell, threshold: float = 0.5) -> None:
        if not isinstance(cell, Cell):
            raise TypeError("Cannot apply SpikingModifier to this type")
        original_compute = cell.compute

        def spiking_compute() -> np.ndarray:
            result = original_compute()
            return np.where(result > threshold, 1, 0)

        cell.compute = spiking_compute