from typing import Optional

from .cell import Cell
from .modifiable import Modifiable

class BackfeedModifier(Modifiable):
    """Adds backfeeding capability to a cell."""

    def __init__(self) -> None:
        self.backfed_value: Optional[np.ndarray] = None

    def apply_modifier(self, cell: Cell) -> None:
        if not isinstance(cell, Cell):
            raise TypeError("Cannot apply BackfeedModifier to this type")
        self.backfed_value = None

        def backfeed(value: np.ndarray) -> None:
            self.backfed_value = value

        def get_backfed() -> Optional[np.ndarray]:
            return self.backfed_value

        cell.backfeed = backfeed
        cell.get_backfed = get_backfed