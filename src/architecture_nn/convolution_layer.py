import numpy as np
from scipy import signal
from typing import Tuple, Callable

from foundational_nn.layer import Layer


class ConvolutionLayer(Layer):
    """Performs convolution operation."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        kernel_size: Tuple[int, int],
        weights: np.ndarray,
        bias: float,
        activation: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(inputs=np.zeros(input_shape))
        self.input_shape: Tuple[int, int] = input_shape
        self.kernel_size: Tuple[int, int] = kernel_size
        self.weights: np.ndarray = weights
        self.bias: float = bias
        self.activation: Callable[[np.ndarray], np.ndarray] = activation

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        conv_result = signal.convolve2d(input_data, self.weights, mode="valid")
        conv_result = conv_result + self.bias
        self.outputs = self.activation(conv_result)
        return self.outputs

    def visualize(self) -> str:
        return "ConvolutionLayer[Output]"
