#!/usr/bin/env python
import numpy as np
from scipy import signal
from typing import List, Callable, Optional, Tuple

# Interfaces
class Visualizable:
    def visualize(self) -> str:
        """Interface for classes that can be visualized."""
        raise NotImplementedError("Subclasses must implement visualize method")

class Modifiable:
    def apply_modifier(self, cell: 'Cell', *args, **kwargs) -> None:
        """Interface for classes that can modify cells."""
        raise NotImplementedError("Subclasses must implement apply_modifier method")

# Abstract Classes
class Cell(Visualizable):
    """
    Abstract class for a basic cell.
    """
    def __init__(self, inputs: Optional[np.ndarray] = None, outputs: Optional[np.ndarray] = None):
        self.inputs: Optional[np.ndarray] = inputs
        self.outputs: Optional[np.ndarray] = outputs

    def compute(self) -> np.ndarray:
        """Computes the cell's output."""
        raise NotImplementedError("Subclasses must implement compute method")

class Layer(Visualizable):
    """
    Abstract class for a layer.
    """
    def __init__(self, inputs: Optional[np.ndarray] = None, outputs: Optional[np.ndarray] = None):
        self.inputs: Optional[np.ndarray] = inputs
        self.outputs: Optional[np.ndarray] = outputs

    def compute(self) -> np.ndarray:
        """Computes the layer's output."""
        raise NotImplementedError("Subclasses must implement compute method")

# Activation Functions
class ActivationFunction:
    """Base class for activation functions."""
    def compute(self, x: np.ndarray) -> np.ndarray:
        """Computes the activation function."""
        raise NotImplementedError("Subclasses must implement compute method")

class SigmoidActivation(ActivationFunction):
    """Sigmoid activation function."""
    def compute(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

class TanhActivation(ActivationFunction):
    """Tanh activation function."""
    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

class ReLUActivation(ActivationFunction):
    """ReLU activation function."""
    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

# Cell Types
class InputCell(Cell):
    """
    Description: Represents a single input value.
    Mathematics: A simple variable.
    Use: Feeding data into the network.
    Architectures: Perceptron, Feed Forward, Radial Basis Network, Deep Feed Forward, Recurrent Neural Network (RNN), Long / Short Term Memory (LSTM), Gated Recurrent Unit (GRU), Auto Encoder (AE), Variational AE (VAE), Denoising AE (DAE), Sparse AE (SAE), Markov Chain (MC), Hopfield Network (HN), Boltzmann Machine (BM), Restricted BM (RBM), Deep Belief Network (DBN), Deep Convolutional Network (DCN), Deconvolutional Network (DN), Deep Convolutional Inverse Graphics Network (DCIGN), Generative Adversarial Network (GAN), Liquid State Machine (LSM), Extreme Learning Machine (ELM), Echo State Network (ESN), Deep Residual Network (DRN), Kohonen Network (KN), Support Vector Machine (SVM), Neural Turing Machine (NTM), Differentiable Neural Computer (DNC), Capsule Network (CN), Attention Network (AN).
    Also used in: Transformers, Diffusion Models
    """
    def __init__(self, value: float):
        super().__init__(inputs=np.array([value]))
        self.value: float = value

    def compute(self) -> np.ndarray:
        self.outputs = np.array([self.value])
        return self.outputs

    def visualize(self) -> str:
        return f"InputCell[{self.outputs}]"

class OutputCell(Cell):
    """
    Description: Produces the final output.
    Mathematics: Weighted sum of inputs + bias, followed by activation.
    Use: Generating predictions, classification, regression, reconstruction, pattern generation.
    Architectures: Perceptron, Feed Forward, Radial Basis Network (RBF), Deep Feed Forward (DFF), Recurrent Neural Network (RNN), Long / Short Term Memory (LSTM), Gated Recurrent Unit (GRU), Auto Encoder (AE), Variational AE (VAE), Denoising AE (DAE), Sparse AE (SAE), Markov Chain (MC), Hopfield Network (HN), Boltzmann Machine (BM), Restricted BM (RBM), Deep Belief Network (DBN), Deep Convolutional Network (DCN), Deconvolutional Network (DN), Deep Convolutional Inverse Graphics Network (DCIGN), Generative Adversarial Network (GAN), Liquid State Machine (LSM), Extreme Learning Machine (ELM), Echo State Network (ESN), Deep Residual Network (DRN), Kohonen Network (KN), Support Vector Machine (SVM), Neural Turing Machine (NTM), Differentiable Neural Computer (DNC), Capsule Network (CN), Attention Network (AN).
    Also used in: Transformers, Diffusion Models
    """
    def __init__(self, inputs: np.ndarray, weights: np.ndarray, bias: float, activation: Callable[[np.ndarray], np.ndarray]):
        super().__init__(inputs=inputs)
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def compute(self) -> np.ndarray:
        weighted_sum = np.dot(self.inputs, self.weights) + self.bias
        self.outputs = self.activation(weighted_sum)
        return self.outputs

    def visualize(self) -> str:
        return f"OutputCell[{self.outputs}]"

class HiddenCell(Cell):
    """
    Description: Intermediate cell for processing data.
    Mathematics: Same as OutputCell.
    Use: Feature extraction, representation learning, non-linear transformations, information processing.
    Architectures: Perceptron (P), Feed Forward (FF), Radial Basis Network (RBF), Deep Feed Forward (DFF), Recurrent Neural Network (RNN), Long / Short Term Memory (LSTM), 
                   Gated Recurrent Unit (GRU), Auto Encoder (AE), Variational AE (VAE), Denoising AE (DAE), Sparse AE (SAE), Deep Belief Network (DBN), Deep Convolutional Network (DCN), Deconvolutional Network (DN)
                   Deep Convolutional Inverse Graphics Network (DCIGN), Generative Adversarial Network (GAN), Liquid State Machine (LSM), Extreme Learning Machine (ELM), Echo State Network (ESN), Deep Residual Network (DRN), Kohonen Network (KN)
                   Support Vector Machine (SVM), Neural Turing Machine (NTM), Differentiable Neural Computer (DNC), Capsule Network (CN), Attention Network (AN).
    Also used in: Transformers, Diffusion Models
    """
    def __init__(self, inputs: np.ndarray, weights: np.ndarray, bias: float, activation: Callable[[np.ndarray], np.ndarray]):
        super().__init__(inputs=inputs)
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def compute(self) -> np.ndarray:
        weighted_sum = np.dot(self.inputs, self.weights) + self.bias
        self.outputs = self.activation(weighted_sum)
        return self.outputs

    def visualize(self) -> str:
        return f"HiddenCell[{self.outputs}]"

class RecurrentCell(Cell):
    """
    Description: Processes sequential data with a feedback loop.
    Mathematics: Hidden state update based on current input and previous hidden state.
    Use: Time series analysis, NLP, speech recognition, sequence generation, dynamic system modeling.
    Architectures: Recurrent Neural Network (RNN), Long / Short Term Memory (LSTM), Gated Recurrent Unit (GRU), Liquid State Machine (LSM), Echo State Network (ESN).
    Also used in: Transformers (in combination with attention)
    """
    def __init__(self, input_t: np.ndarray, hidden_state_t_1: np.ndarray, weight_in: np.ndarray,
                 weight_hidden: np.ndarray, bias: float, activation: Callable[[np.ndarray], np.ndarray]):
        super().__init__(inputs=[input_t, hidden_state_t_1])
        self.input_t = input_t
        self.hidden_state_t_1 = hidden_state_t_1
        self.weight_in = weight_in
        self.weight_hidden = weight_hidden
        self.bias = bias
        self.activation = activation

    def compute(self) -> np.ndarray:
        new_hidden_state = self.activation(np.dot(self.input_t, self.weight_in) +
                                            np.dot(self.hidden_state_t_1, self.weight_hidden) +
                                            self.bias)
        self.outputs = new_hidden_state
        return self.outputs

    def visualize(self) -> str:
        return f"RecurrentCell[{self.outputs}]"

class MemoryCell(Cell):
    """
    Description: Core memory unit for LSTM-like cells.
    Mathematics: Cell state update equation.
    Use: Storing information over time, capturing long-range dependencies, sequence modeling, state representation.
    Architectures: Long / Short Term Memory (LSTM), Gated Recurrent Unit (GRU), Neural Turing Machine (NTM), Differentiable Neural Computer (DNC).
    """
    def __init__(self, forget_gate_output: np.ndarray, input_gate_output: np.ndarray,
                 cell_gate_output: np.ndarray, previous_cell_state: np.ndarray):
        super().__init__(inputs=[forget_gate_output, input_gate_output, cell_gate_output, previous_cell_state])
        self.forget_gate_output = forget_gate_output
        self.input_gate_output = input_gate_output
        self.cell_gate_output = cell_gate_output
        self.previous_cell_state = previous_cell_state

    def compute(self) -> np.ndarray:
        new_cell_state = self.forget_gate_output * self.previous_cell_state + self.input_gate_output * self.cell_gate_output
        self.outputs = new_cell_state
        return self.outputs

    def visualize(self) -> str:
        return f"MemoryCell[{self.outputs}]"

class GatedMemoryCell(Cell):
    """
    Description: Gated memory cell (generalization of LSTM's memory cell).
    Mathematics: Defined by the gate_logic function.
    Use: More flexible memory control, complex sequential processing, attention mechanisms, dynamic state management.
    Architectures: Advanced RNNs, Transformers (as a form of gated attention).
    """
    def __init__(self, gates_output: np.ndarray, cell_state_t_1: np.ndarray):
        super().__init__(inputs=[gates_output, cell_state_t_1])
        self.gates_output = gates_output
        self.cell_state_t_1 = cell_state_t_1

    def compute(self, gate_logic: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        self.outputs = gate_logic(self.gates_output, self.cell_state_t_1)
        return self.outputs

    def visualize(self) -> str:
        return f"GatedMemoryCell[{self.outputs}]"

class Gate(Cell):
    """
    Description: Gating mechanism (LSTM-like).
    Mathematics: Sigmoid activation.
    Use: Controlling information flow, attention mechanisms, modulation of signals.
    Architectures: Long / Short Term Memory (LSTM), Gated Recurrent Unit (GRU).
    Also used in: Transformers (gating mechanisms in some variations)
    """
    def __init__(self, inputs: np.ndarray, weights: np.ndarray, bias: float, activation: Callable[[np.ndarray], np.ndarray]):
        super().__init__(inputs=inputs)
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def compute(self) -> np.ndarray:
        self.outputs = self.activation(np.dot(self.inputs, self.weights) + self.bias)
        return self.outputs

    def visualize(self) -> str:
        return f"Gate[{self.outputs}]"

class CapsuleCell(Cell):
    """
    Description: A cell type that represents entities as vectors ("capsules").
    Mathematics: Applies a transformation matrix. Routing is often performed outside the cell definition.
    Use: Capturing hierarchical relationships in data, object recognition, scene understanding, part-whole relationships.
    Architectures: Capsule Network (CN).
    """
    def __init__(self, inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray,
                 activation: Callable[[np.ndarray], np.ndarray],
                 routing_iterations: int):
        super().__init__(inputs=inputs)
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.routing_iterations = routing_iterations

    def compute(self) -> np.ndarray:
        # Simple capsule computation without iterative routing.
        weighted_sum = np.dot(self.inputs, self.weights) + self.bias
        self.outputs = self.activation(weighted_sum)
        return self.outputs

    def visualize(self) -> str:
        return f"CapsuleCell[{self.outputs}]"

class ConvolutionLayer(Layer):
    """
    Description: Performs convolution operation.
    Mathematics: 2D convolution of input and kernel.
    Use: Feature extraction in images, signal processing, pattern recognition.
    Architectures: Deep Convolutional Network (DCN), Deconvolutional Network (DN), Deep Convolutional Inverse Graphics Network (DCIGN), Generative Adversarial Network (GAN).
    """
    def __init__(self, input_shape: Tuple[int, int], kernel_size: Tuple[int, int],
                 weights: np.ndarray, bias: float, activation: Callable[[np.ndarray], np.ndarray]):
        # input_shape is (height, width)
        super().__init__(inputs=np.zeros(input_shape))
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.weights = weights  # Expecting a 2D kernel
        self.bias = bias        # Bias as a scalar for simplicity
        self.activation = activation

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        # Using valid mode convolution from scipy.signal
        conv_result = signal.convolve2d(input_data, self.weights, mode='valid')
        conv_result = conv_result + self.bias
        self.outputs = self.activation(conv_result)
        return self.outputs

    def visualize(self) -> str:
        return "ConvolutionLayer[Output]"

class PoolingLayer(Layer):
    """
    Description: Performs pooling operation.
    Mathematics: Max or Average pooling.
    Use: Downsampling feature maps, reducing dimensionality, translation invariance.
    Architectures: Deep Convolutional Network (DCN), Deconvolutional Network (DN), Deep Convolutional Inverse Graphics Network (DCIGN).
    """
    def __init__(self, input_shape: Tuple[int, int], pool_size: Tuple[int, int],
                 stride: Tuple[int, int], operation: str = 'max'):
        super().__init__(inputs=np.zeros(input_shape))
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = stride
        self.operation = operation

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        # Calculate output dimensions
        h, w = input_data.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride
        out_h = (h - pool_h) // stride_h + 1
        out_w = (w - pool_w) // stride_w + 1
        output = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                patch = input_data[i*stride_h:i*stride_h+pool_h, j*stride_w:j*stride_w+pool_w]
                if self.operation == 'max':
                    output[i, j] = np.max(patch)
                else:
                    output[i, j] = np.mean(patch)
        self.outputs = output
        return self.outputs

    def visualize(self) -> str:
        return "PoolingLayer[Output]"

class Kernel(Cell):
    """
    Description: Extracts features from input data using convolution or pooling.
    """
    def __init__(self, kernel_matrix: Optional[np.ndarray] = None):
        super().__init__(inputs=kernel_matrix)
        self.kernel_matrix = kernel_matrix

    def compute(self, input_data: np.ndarray, operation: str = 'convolve') -> np.ndarray:
        if operation == 'convolve':
            if self.kernel_matrix is None:
                raise ValueError("Kernel matrix is required for convolution")
            # For simplicity, use the kernel_matrix directly as the convolution filter.
            conv_layer = ConvolutionLayer(input_shape=input_data.shape,
                                          kernel_size=self.kernel_matrix.shape,
                                          weights=self.kernel_matrix,
                                          bias=0.0,  # using zero bias
                                          activation=ReLUActivation().compute)
            self.outputs = conv_layer.compute(input_data)
        elif operation == 'pool':
            pool_layer = PoolingLayer(input_shape=input_data.shape,
                                      pool_size=(2, 2),
                                      stride=(2, 2),
                                      operation='max')
            self.outputs = pool_layer.compute(input_data)
        else:
            raise ValueError("Invalid operation: {}".format(operation))
        return self.outputs

    def visualize(self) -> str:
        return f"Kernel[{self.outputs}]"

class MatchInputOutput(Cell):
    """
    Description: Used in autoencoders for reconstruction.
    Mathematics: Matrix multiplication.
    Use: Reconstruction of input data.
    """
    def __init__(self, input_data: np.ndarray, encoder_weights: np.ndarray,
                 decoder_weights: np.ndarray):
        super().__init__(inputs=input_data)
        self.input_data = input_data
        self.encoder_weights = encoder_weights
        self.decoder_weights = decoder_weights

    def compute(self) -> np.ndarray:
        encoded = np.dot(self.input_data, self.encoder_weights)
        self.outputs = np.dot(encoded, self.decoder_weights)
        return self.outputs

    def visualize(self) -> str:
        return f"MatchInputOutput[{self.outputs}]"

# Modifiers
class NoisyModifier(Modifiable):
    """Adds noise to cell output."""
    def apply_modifier(self, cell: Cell, noise_std: float = 1.0) -> None:
        if isinstance(cell, Cell):
            original_compute = cell.compute
            def noisy_compute() -> np.ndarray:
                result = original_compute()
                return result + np.random.normal(0, noise_std, size=result.shape)
            cell.compute = noisy_compute
        else:
            raise TypeError("Cannot apply NoisyModifier to this type")

class ProbabilisticModifier(Modifiable):
    """Makes cell output probabilistic."""
    def apply_modifier(self, cell: Cell, mean: float = 0.0, std_dev: float = 1.0) -> None:
        if isinstance(cell, Cell):
            original_compute = cell.compute
            def probabilistic_compute() -> np.ndarray:
                _ = original_compute()  # Call original_compute to get shape, if needed.
                return np.random.normal(mean, std_dev, size=cell.outputs.shape)
            cell.compute = probabilistic_compute
        else:
            raise TypeError("Cannot apply ProbabilisticModifier to this type")

class SpikingModifier(Modifiable):
    """Makes cell behave like a spiking neuron."""
    def apply_modifier(self, cell: Cell, threshold: float = 0.5) -> None:
        if isinstance(cell, Cell):
            original_compute = cell.compute
            def spiking_compute() -> np.ndarray:
                result = original_compute()
                return np.where(result > threshold, 1, 0)
            cell.compute = spiking_compute
        else:
            raise TypeError("Cannot apply SpikingModifier to this type")

class BackfeedModifier(Modifiable):
    """Adds backfeeding capability to a cell."""
    def __init__(self):
        self.backfed_value: Optional[np.ndarray] = None

    def apply_modifier(self, cell: Cell) -> None:
        if isinstance(cell, Cell):
            self.backfed_value = None
            def backfeed(value: np.ndarray) -> None:
                self.backfed_value = value
            def get_backfed() -> Optional[np.ndarray]:
                return self.backfed_value
            cell.backfeed = backfeed
            cell.get_backfed = get_backfed
        else:
            raise TypeError("Cannot apply BackfeedModifier to this type")

# Atomic Examples
class AtomicExamples:
    def __init__(self):
        pass

    def input_cell_example(self) -> str:
        input_cell = InputCell(5.0)
        input_cell.compute()
        return input_cell.visualize()

    def output_cell_example(self) -> str:
        output_cell = OutputCell(inputs=np.array([2.0, 3.0]), weights=np.array([0.5, 0.3]),
                                 bias=0.1, activation=ReLUActivation().compute)
        output_cell.compute()
        return output_cell.visualize()

    def hidden_cell_example(self) -> str:
        hidden_cell = HiddenCell(inputs=np.array([2.0, 3.0]), weights=np.array([0.5, 0.3]),
                                 bias=0.1, activation=TanhActivation().compute)
        hidden_cell.compute()
        return hidden_cell.visualize()

    def recurrent_cell_example(self) -> str:
        recurrent_cell = RecurrentCell(input_t=np.array([1.0]),
                                       hidden_state_t_1=np.array([0.5]),
                                       weight_in=np.array([0.4]),
                                       weight_hidden=np.array([0.6]),
                                       bias=0.2,
                                       activation=TanhActivation().compute)
        recurrent_cell.compute()
        return recurrent_cell.visualize()

    def memory_cell_example(self) -> str:
        memory_cell = MemoryCell(forget_gate_output=np.array([0.8]),
                                 input_gate_output=np.array([0.9]),
                                 cell_gate_output=np.array([0.1]),
                                 previous_cell_state=np.array([0.5]))
        memory_cell.compute()
        return memory_cell.visualize()

    def gated_memory_cell_example(self) -> np.ndarray:
        def simple_gate_logic(gates_output: np.ndarray, cell_state_t_1: np.ndarray) -> np.ndarray:
            return gates_output * cell_state_t_1
        gated_memory_cell = GatedMemoryCell(gates_output=np.array([0.5, 0.6]),
                                            cell_state_t_1=np.array([0.2, 0.2]))
        gated_memory_cell.compute(gate_logic=simple_gate_logic)
        return gated_memory_cell.outputs

    def gate_example(self) -> str:
        gate = Gate(inputs=np.array([1.0, 2.0]),
                    weights=np.array([0.5, 0.5]),
                    bias=0.1,
                    activation=SigmoidActivation().compute)
        gate.compute()
        return gate.visualize()

    def capsule_cell_example(self) -> str:
        capsule_cell = CapsuleCell(inputs=np.array([1.0, 2.0]),
                                   weights=np.array([[0.5, 0.5], [0.5, 0.5]]),
                                   bias=np.array([0.1, 0.1]),
                                   activation=ReLUActivation().compute,
                                   routing_iterations=3)
        capsule_cell.compute()
        return capsule_cell.visualize()

    def kernel_convolve_example(self) -> np.ndarray:
        kernel = Kernel(kernel_matrix=np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
        return kernel.compute(input_data=np.array([[1, 2, 3],
                                                   [4, 5, 6],
                                                   [7, 8, 9]]), operation='convolve')

    def kernel_pool_example(self) -> np.ndarray:
        kernel = Kernel(kernel_matrix=None)
        return kernel.compute(input_data=np.array([[1, 2, 3, 4],
                                                   [5, 6, 7, 8],
                                                   [9, 10, 11, 12],
                                                   [13, 14, 15, 16]]), operation='pool')

    def noisy_modifier_example(self) -> np.ndarray:
        input_cell = InputCell(5.0)
        noisy_input_cell = InputCell(5.0)
        NoisyModifier().apply_modifier(noisy_input_cell, noise_std=0.5)
        return noisy_input_cell.compute()

    def probabilistic_modifier_example(self) -> np.ndarray:
        hidden_cell = HiddenCell(inputs=np.array([2.0, 3.0]),
                                 weights=np.array([0.5, 0.3]),
                                 bias=0.1,
                                 activation=SigmoidActivation().compute)
        prob_hidden_cell = HiddenCell(inputs=np.array([2.0, 3.0]),
                                      weights=np.array([0.5, 0.3]),
                                      bias=0.1,
                                      activation=SigmoidActivation().compute)
        ProbabilisticModifier().apply_modifier(prob_hidden_cell, mean=0.0, std_dev=1.0)
        return prob_hidden_cell.compute()

    def spiking_modifier_example(self) -> np.ndarray:
        hidden_cell = HiddenCell(inputs=np.array([2.0, 3.0]),
                                 weights=np.array([0.5, 0.3]),
                                 bias=0.1,
                                 activation=SigmoidActivation().compute)
        spiking_hidden_cell = HiddenCell(inputs=np.array([2.0, 3.0]),
                                         weights=np.array([0.5, 0.3]),
                                         bias=0.1,
                                         activation=SigmoidActivation().compute)
        SpikingModifier().apply_modifier(spiking_hidden_cell, threshold=0.5)
        return spiking_hidden_cell.compute()

    def backfeed_modifier_example(self) -> Optional[np.ndarray]:
        input_cell = InputCell(5.0)
        backfed_input_cell = InputCell(5.0)
        BackfeedModifier().apply_modifier(backfed_input_cell)
        backfed_input_cell.backfeed(np.array([10.0]))
        return backfed_input_cell.get_backfed()

def main():
    examples = AtomicExamples()
    print("Input Cell Example:", examples.input_cell_example())
    print("Output Cell Example:", examples.output_cell_example())
    print("Hidden Cell Example:", examples.hidden_cell_example())
    print("Recurrent Cell Example:", examples.recurrent_cell_example())
    print("Memory Cell Example:", examples.memory_cell_example())
    print("Gated Memory Cell Example:", examples.gated_memory_cell_example())
    print("Gate Example:", examples.gate_example())
    print("Capsule Cell Example:", examples.capsule_cell_example())
    print("Kernel Convolve Example:\n", examples.kernel_convolve_example())
    print("Kernel Pool Example:\n", examples.kernel_pool_example())
    print("Noisy Modifier Example:\n", examples.noisy_modifier_example())
    print("Probabilistic Modifier Example:\n", examples.probabilistic_modifier_example())
    print("Spiking Modifier Example:\n", examples.spiking_modifier_example())
    print("Backfeed Modifier Example:\n", examples.backfeed_modifier_example())

if __name__ == "__main__":
    main()
