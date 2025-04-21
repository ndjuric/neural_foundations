from typing import List
import numpy as np

from foundational_nn.layer import Layer
from foundational_nn.visualizable import Visualizable


class SequentialModel(Visualizable):
    """Sequential model composed of layers."""

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def compute(self, input_data: np.ndarray) -> np.ndarray:
        data = input_data
        for layer in self.layers:
            data = layer.compute(data)
        return data

    def visualize(self) -> str:
        visualization = "SequentialModel\n"
        for layer in self.layers:
            visualization += f"{layer.visualize()}\n"
        return visualization

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the model."""
        self.layers.append(layer)

    def remove_layer(self, index: int) -> None:
        """Remove a layer from the model by index."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)