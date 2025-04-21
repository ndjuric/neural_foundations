import numpy as np
from typing import Optional

from .input_cell import InputCell
from .output_cell import OutputCell
from .hidden_cell import HiddenCell
from .recurrent_cell import RecurrentCell
from .memory_cell import MemoryCell
from .gated_memory_cell import GatedMemoryCell
from .gate import Gate
from .capsule_cell import CapsuleCell
from .kernel import Kernel
from .noisy_modifier import NoisyModifier
from .probabilistic_modifier import ProbabilisticModifier
from .spiking_modifier import SpikingModifier
from .backfeed_modifier import BackfeedModifier
from .sigmoid_activation import SigmoidActivation
from .tanh_activation import TanhActivation
from .relu_activation import ReLUActivation

class AtomicExamples:
    """Demonstrate atomic neural network primitives."""

    def input_cell_example(self) -> str:
        input_cell = InputCell(5.0)
        input_cell.compute()
        return input_cell.visualize()

    def output_cell_example(self) -> str:
        output_cell = OutputCell(inputs=np.array([2.0, 3.0]), weights=np.array([0.5, 0.3]), bias=0.1, activation=ReLUActivation().compute)
        output_cell.compute()
        return output_cell.visualize()

    def hidden_cell_example(self) -> str:
        hidden_cell = HiddenCell(inputs=np.array([2.0, 3.0]), weights=np.array([0.5, 0.3]), bias=0.1, activation=TanhActivation().compute)
        hidden_cell.compute()
        return hidden_cell.visualize()

    def recurrent_cell_example(self) -> str:
        recurrent_cell = RecurrentCell(input_t=np.array([1.0]), hidden_state_t_1=np.array([0.5]), weight_in=np.array([0.4]), weight_hidden=np.array([0.6]), bias=0.2, activation=TanhActivation().compute)
        recurrent_cell.compute()
        return recurrent_cell.visualize()

    def memory_cell_example(self) -> str:
        memory_cell = MemoryCell(forget_gate_output=np.array([0.8]), input_gate_output=np.array([0.9]), cell_gate_output=np.array([0.1]), previous_cell_state=np.array([0.5]))
        memory_cell.compute()
        return memory_cell.visualize()

    def gated_memory_cell_example(self) -> np.ndarray:
        def simple_gate_logic(gates_output: np.ndarray, cell_state_t_1: np.ndarray) -> np.ndarray:
            return gates_output * cell_state_t_1

        gated_memory_cell = GatedMemoryCell(gates_output=np.array([0.5, 0.6]), cell_state_t_1=np.array([0.2, 0.2]))
        gated_memory_cell.compute(gate_logic=simple_gate_logic)
        return gated_memory_cell.outputs

    def gate_example(self) -> str:
        gate = Gate(inputs=np.array([1.0, 2.0]), weights=np.array([0.5, 0.5]), bias=0.1, activation=SigmoidActivation().compute)
        gate.compute()
        return gate.visualize()

    def capsule_cell_example(self) -> str:
        capsule_cell = CapsuleCell(inputs=np.array([1.0, 2.0]), weights=np.array([[0.5, 0.5], [0.5, 0.5]]), bias=np.array([0.1, 0.1]), activation=ReLUActivation().compute, routing_iterations=3)
        capsule_cell.compute()
        return capsule_cell.visualize()

    def kernel_convolve_example(self) -> np.ndarray:
        kernel = Kernel(kernel_matrix=np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
        return kernel.compute(input_data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), operation='convolve')

    def kernel_pool_example(self) -> np.ndarray:
        kernel = Kernel(kernel_matrix=None)
        return kernel.compute(input_data=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), operation='pool')

    def noisy_modifier_example(self) -> np.ndarray:
        noisy_input_cell = InputCell(5.0)
        NoisyModifier().apply_modifier(noisy_input_cell, noise_std=0.5)
        return noisy_input_cell.compute()

    def probabilistic_modifier_example(self) -> np.ndarray:
        prob_hidden_cell = HiddenCell(inputs=np.array([2.0, 3.0]), weights=np.array([0.5, 0.3]), bias=0.1, activation=SigmoidActivation().compute)
        ProbabilisticModifier().apply_modifier(prob_hidden_cell, mean=0.0, std_dev=1.0)
        return prob_hidden_cell.compute()

    def spiking_modifier_example(self) -> np.ndarray:
        spiking_hidden_cell = HiddenCell(inputs=np.array([2.0, 3.0]), weights=np.array([0.5, 0.3]), bias=0.1, activation=SigmoidActivation().compute)
        SpikingModifier().apply_modifier(spiking_hidden_cell, threshold=0.5)
        return spiking_hidden_cell.compute()

    def backfeed_modifier_example(self) -> Optional[np.ndarray]:
        backfed_input_cell = InputCell(5.0)
        modifier = BackfeedModifier()
        modifier.apply_modifier(backfed_input_cell)
        backfed_input_cell.backfeed(np.array([10.0]))
        return backfed_input_cell.get_backfed()

def main() -> None:
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