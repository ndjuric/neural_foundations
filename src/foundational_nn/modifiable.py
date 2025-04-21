from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cell import Cell

class Modifiable(ABC):
    """Interface for classes that can modify cells."""

    @abstractmethod
    def apply_modifier(self, cell: 'Cell', *args, **kwargs) -> None:
        """Apply a modifier to a cell."""
        pass