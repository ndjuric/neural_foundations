"""Architectural patterns built using foundational and layer primitives."""
import numpy as np
from typing import List, Tuple, Optional, Callable

from foundational_nn.visualizable import Visualizable
from foundational_nn.match_input_output import MatchInputOutput
from architecture_nn.dense_layer import DenseLayer
from architecture_nn.sequential_model import SequentialModel
from architecture_nn.recurrent_layer import RecurrentLayer
from architecture_nn.convolution_layer import ConvolutionLayer
from architecture_nn.pooling_layer import PoolingLayer


class FeedForwardNetwork(Visualizable):
    """Feed-forward network pattern using dense layers."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation_hidden: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        activation_output: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        from foundational_nn.relu_activation import ReLUActivation
        from foundational_nn.sigmoid_activation import SigmoidActivation

        if activation_hidden is None:
            activation_hidden = ReLUActivation().compute
        if activation_output is None:
            activation_output = SigmoidActivation().compute

        layers: List[DenseLayer] = []
        last_size = input_size
        for h in hidden_sizes:
            layers.append(
                DenseLayer(
                    input_size=last_size,
                    output_size=h,
                    activation=activation_hidden,
                )
            )
            last_size = h
        layers.append(
            DenseLayer(
                input_size=last_size,
                output_size=output_size,
                activation=activation_output,
            )
        )
        self.model = SequentialModel(layers)

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        return self.model.compute(input_data)

    def visualize(self) -> str:
        return self.model.visualize()


class RNN(Visualizable):
    """Recurrent neural network pattern."""

    def __init__(
        self,
        weight_in: np.ndarray,
        weight_hidden: np.ndarray,
        bias: float,
        activation: Callable[[np.ndarray], np.ndarray],
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        self.layer = RecurrentLayer(
            weight_in=weight_in,
            weight_hidden=weight_hidden,
            bias=bias,
            activation=activation,
            initial_state=initial_state,
        )

    def compute(self, input_sequence: np.ndarray) -> np.ndarray:
        return self.layer.compute(input_sequence)

    def visualize(self) -> str:
        return self.layer.visualize()


class CNN(Visualizable):
    """Convolutional neural network pattern."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        conv_kernels: List[np.ndarray],
        conv_biases: List[float],
        pool_sizes: List[Tuple[int, int]],
        pool_strides: List[Tuple[int, int]],
        hidden_sizes: List[int],
        num_classes: int,
        activation_conv: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        activation_hidden: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        activation_output: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        from foundational_nn.relu_activation import ReLUActivation
        from foundational_nn.sigmoid_activation import SigmoidActivation

        if activation_conv is None:
            activation_conv = ReLUActivation().compute
        if activation_hidden is None:
            activation_hidden = ReLUActivation().compute
        if activation_output is None:
            activation_output = SigmoidActivation().compute

        layers = []
        last_shape = input_shape
        # convolution and pooling stages
        for kernel, bias, pool_size, pool_stride in zip(
            conv_kernels, conv_biases, pool_sizes, pool_strides
        ):
            layers.append(
                ConvolutionLayer(
                    input_shape=last_shape,
                    kernel_size=kernel.shape,
                    weights=kernel,
                    bias=bias,
                    activation=activation_conv,
                )
            )
            # update shape after convolution
            new_h = last_shape[0] - kernel.shape[0] + 1
            new_w = last_shape[1] - kernel.shape[1] + 1
            last_shape = (new_h, new_w)
            layers.append(
                PoolingLayer(
                    input_shape=last_shape,
                    pool_size=pool_size,
                    stride=pool_stride,
                    operation="max",
                )
            )
            # update shape after pooling
            last_shape = (
                (last_shape[0] - pool_size[0]) // pool_stride[0] + 1,
                (last_shape[1] - pool_size[1]) // pool_stride[1] + 1,
            )
        # flatten layer
        class FlattenLayer:
            def __init__(self, shape: Tuple[int, int]):
                self.inputs = None
                self.outputs = None
                self.shape = shape

            def compute(self, input_data: np.ndarray) -> np.ndarray:
                self.inputs = input_data
                self.outputs = input_data.flatten()
                return self.outputs

            def visualize(self) -> str:
                return f"FlattenLayer[{self.shape}]"

        layers.append(FlattenLayer(last_shape))
        # dense classification stages
        last_size = last_shape[0] * last_shape[1]
        for h in hidden_sizes:
            layers.append(
                DenseLayer(
                    input_size=last_size,
                    output_size=h,
                    activation=activation_hidden,
                )
            )
            last_size = h
        layers.append(
            DenseLayer(
                input_size=last_size,
                output_size=num_classes,
                activation=activation_output,
            )
        )
        self.model = SequentialModel(layers)

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        return self.model.compute(input_data)

    def visualize(self) -> str:
        return self.model.visualize()


class AutoEncoder(Visualizable):
    """Autoencoder pattern for encoding and reconstructing data."""

    def __init__(
        self,
        input_size: int,
        encoding_dim: int,
    ) -> None:
        # initialize weights
        self.encoder_weights = np.random.rand(input_size, encoding_dim)
        self.decoder_weights = np.random.rand(encoding_dim, input_size)
        # underlying model
        self.model = MatchInputOutput(
            input_data=None,
            encoder_weights=self.encoder_weights,
            decoder_weights=self.decoder_weights,
        )

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        # set input and compute reconstruction
        self.model.input_data = input_data
        return self.model.compute()

    def visualize(self) -> str:
        return self.model.visualize()