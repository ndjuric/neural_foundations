"""Recurrent layer composed of recurrent cells."""
from typing import Optional, List, Callable
import numpy as np

from foundational_nn.recurrent_cell import RecurrentCell
from foundational_nn.layer import Layer


class RecurrentLayer(Layer):
    """Layer for processing sequences with recurrent cells."""

    def __init__(
        self,
        weight_in: np.ndarray,
        weight_hidden: np.ndarray,
        bias: float,
        activation: Callable[[np.ndarray], np.ndarray],
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(inputs=None)
        self.weight_in = weight_in
        self.weight_hidden = weight_hidden
        self.bias = bias
        self.activation = activation
        # initialize hidden state
        if initial_state is None:
            # zero vector same size as weight_hidden
            self.hidden_state = np.zeros_like(weight_hidden)
        else:
            self.hidden_state = initial_state

    def compute(self, input_sequence: np.ndarray) -> np.ndarray:
        """Compute outputs for a sequence of inputs."""
        self.inputs = input_sequence
        outputs: List[np.ndarray] = []
        state = self.hidden_state
        for x in input_sequence:
            cell = RecurrentCell(
                input_t=x,
                hidden_state_t_1=state,
                weight_in=self.weight_in,
                weight_hidden=self.weight_hidden,
                bias=self.bias,
                activation=self.activation,
            )
            state = cell.compute()
            outputs.append(state)
        self.hidden_state = state
        self.outputs = np.array(outputs)
        return self.outputs

    def visualize(self) -> str:
        return f"RecurrentLayer[hidden_state={self.hidden_state}]"