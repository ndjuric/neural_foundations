#!/usr/bin/env python
from typing import Optional

class Network:
    """
    Simple graph builder for neural architectures.
    Use add_node(), connect(), then visualize() to get a Mermaid graph string.
    """
    def __init__(self):
        self._nodes = {}  # node_id -> label
        self._edges = []  # list of (src_id, dst_id)

    def add_node(self, node_id: str, label: Optional[str] = None):
        """
        Register a node in the network.
        node_id: unique identifier string.
        label: display label for the node (defaults to node_id).
        """
        self._nodes[node_id] = label or node_id

    def connect(self, src: str, dst: str):
        """
        Add a directed edge from src to dst.
        """
        if src not in self._nodes or dst not in self._nodes:
            raise ValueError(f"Unknown node '{src}' or '{dst}'")
        self._edges.append((src, dst))

    def visualize(self) -> str:
        """
        Generate a Mermaid 'graph LR' representation of the network.
        """
        lines = ["graph LR"]
        for node_id, label in self._nodes.items():
            lines.append(f"    {node_id}[\"{label}\"]")
        for src, dst in self._edges:
            lines.append(f"    {src} --> {dst}")
        return "\n".join(lines)