import numpy as np
from typing import Tuple

from foundational_nn.layer import Layer


class PoolingLayer(Layer):
    """Performs pooling operation."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        pool_size: Tuple[int, int],
        stride: Tuple[int, int],
        operation: str = "max",
    ) -> None:
        super().__init__(inputs=np.zeros(input_shape))
        self.input_shape: Tuple[int, int] = input_shape
        self.pool_size: Tuple[int, int] = pool_size
        self.stride: Tuple[int, int] = stride
        self.operation: str = operation

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        h, w = input_data.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride
        out_h = (h - pool_h) // stride_h + 1
        out_w = (w - pool_w) // stride_w + 1
        output = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                patch = input_data[
                    i * stride_h : i * stride_h + pool_h,
                    j * stride_w : j * stride_w + pool_w,
                ]
                if self.operation == "max":
                    output[i, j] = np.max(patch)
                else:
                    output[i, j] = np.mean(patch)
        self.outputs = output
        return self.outputs

    def visualize(self) -> str:
        return "PoolingLayer[Output]"
