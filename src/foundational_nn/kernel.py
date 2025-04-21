import numpy as np

from typing import Optional

from .convolution_layer import ConvolutionLayer
from .pooling_layer import PoolingLayer
from .relu_activation import ReLUActivation

from .cell import Cell

class Kernel(Cell):
    """Extracts features from input data using convolution or pooling."""

    def __init__(self, kernel_matrix: Optional[np.ndarray] = None) -> None:
        super().__init__(inputs=kernel_matrix)
        self.kernel_matrix: Optional[np.ndarray] = kernel_matrix

    def compute(self, input_data: np.ndarray, operation: str = 'convolve') -> np.ndarray:
        if operation == 'convolve':
            if self.kernel_matrix is None:
                raise ValueError("Kernel matrix is required for convolution")
            conv_layer = ConvolutionLayer(
                input_shape=input_data.shape,
                kernel_size=self.kernel_matrix.shape,
                weights=self.kernel_matrix,
                bias=0.0,
                activation=ReLUActivation().compute
            )
            self.outputs = conv_layer.compute(input_data)
        elif operation == 'pool':
            pool_layer = PoolingLayer(
                input_shape=input_data.shape,
                pool_size=(2, 2),
                stride=(2, 2),
                operation='max'
            )
            self.outputs = pool_layer.compute(input_data)
        else:
            raise ValueError(f"Invalid operation: {operation}")
        return self.outputs

    def visualize(self) -> str:
        return f"Kernel[{self.outputs}]"