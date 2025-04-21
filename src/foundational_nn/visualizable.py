from abc import ABC, abstractmethod

class Visualizable(ABC):
    """Interface for classes that can be visualized."""

    @abstractmethod
    def visualize(self) -> str:
        """Return a string representation for visualization."""
        pass