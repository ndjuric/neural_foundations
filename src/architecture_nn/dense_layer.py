import numpy as np
from typing import Callable

from foundational_nn.hidden_cell import HiddenCell
from foundational_nn.output_cell import OutputCell
from foundational_nn.layer import Layer


class DenseLayer(Layer):
    """Dense layer composed of multiple hidden or output cells."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(inputs=np.zeros(input_size))
        self.cells = [
            HiddenCell(
                inputs=np.zeros(input_size),
                weights=np.random.rand(input_size),
                bias=np.random.rand(),
                activation=activation,
            )
            for _ in range(output_size)
        ]

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        self.inputs = input_data
        outputs = np.array([cell.compute() for cell in self.cells])
        self.outputs = outputs
        return self.outputs

    def visualize(self) -> str:
        return "DenseLayer[Output]"