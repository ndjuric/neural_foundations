"""Graph-based model builder for arbitrary layer connections."""
from typing import Dict, List, Any
import numpy as np

from foundational_nn.layer import Layer
from foundational_nn.visualizable import Visualizable
from architecture_nn.network import Network


class GraphModel(Visualizable):
    """Model defined as a directed graph of layers."""

    def __init__(self) -> None:
        self.network = Network()
        self.layers: Dict[str, Layer] = {}

    def add_layer(self, node_id: str, layer: Layer) -> None:
        """Register a layer node in the graph."""
        self.network.add_node(node_id)
        self.layers[node_id] = layer

    def connect(self, src: str, dst: str) -> None:
        """Add a directed connection from src to dst."""
        self.network.connect(src, dst)

    def compute(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute outputs for all layers given initial inputs."""
        data: Dict[str, np.ndarray] = {}
        # set provided inputs
        for node_id, value in inputs.items():
            data[node_id] = value

        # compute in topological order (insertion order)
        for node_id in self.network._nodes:
            if node_id in data:
                continue
            # gather predecessors
            preds = [src for (src, dst) in self.network._edges if dst == node_id]
            if not preds:
                continue
            # sum inputs from predecessors
            inputs_list: List[np.ndarray] = [data[p] for p in preds]
            x = sum(inputs_list)
            # compute layer output
            layer = self.layers[node_id]
            data[node_id] = layer.compute(x)
        return data

    def visualize(self) -> str:
        return self.network.visualize()