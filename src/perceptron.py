#!/usr/bin/env python
import random

from foundational_nn.output_cell import OutputCell
from foundational_nn.sigmoid_activation import SigmoidActivation
from network import Network


class Perceptron:
    """
    Simple binary classifier using atomic primitives.
    """
    def __init__(self, n_inputs: int, learning_rate: float = 1.0):
        self.n_inputs = n_inputs
        self.lr = learning_rate
        # initialize weights and bias
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.activation = SigmoidActivation().compute

    def predict(self, x):
        """
        Compute binary prediction for input vector x.
        """
        cell = OutputCell(inputs=x,
                          weights=self.weights,
                          bias=self.bias,
                          activation=self.activation)
        out = cell.compute()
        # out may be array or scalar
        val = out[0] if hasattr(out, "__iter__") else out
        return 1 if val >= 0.5 else 0

    def train(self, X, y, epochs: int = 10):
        """
        Train on dataset (X, y) for a number of epochs.
        X: iterable of input vectors (lists of floats)
        y: iterable of target labels (0 or 1)
        """
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                y_hat = self.predict(xi)
                error = yi - y_hat
                # update weights and bias
                for i, v in enumerate(xi):
                    self.weights[i] += self.lr * error * v
                self.bias += self.lr * error

    def visualize(self) -> str:
        """
        Return a Mermaid diagram of the perceptron's structure.
        """
        net = Network()
        # add inputs
        for i in range(self.n_inputs):
            net.add_node(f"in{i}", f"In{i}")
        # add output
        net.add_node("out", "Out")
        # connect inputs to output
        for i in range(self.n_inputs):
            net.connect(f"in{i}", "out")
        return net.visualize()
