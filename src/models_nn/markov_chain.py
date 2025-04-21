#!/usr/bin/env python
import random

from architecture_nn.network import Network


class MarkovChain:
    """
    Simple character-level Markov chain for text generation using atomic primitives.
    """

    def __init__(self, k: int = 1):
        self.k = k
        self.model = {}

    def train(self, text: str):
        """
        Build k-gram model from text.
        """
        for i in range(len(text) - self.k):
            key = text[i : i + self.k]
            nxt = text[i + self.k]
            self.model.setdefault(key, {})
            self.model[key][nxt] = self.model[key].get(nxt, 0) + 1

    def generate(self, length: int = 100, seed: str = None) -> str:
        """
        Generate text of given length from the model.
        """
        if not self.model:
            return ""
        if seed is None or len(seed) != self.k or seed not in self.model:
            seed = next(iter(self.model))
        result = seed
        for _ in range(length - len(seed)):
            key = result[-self.k :]
            choices = self.model.get(key, {})
            if not choices:
                break
            chars, counts = zip(*choices.items())
            total = sum(counts)
            # choose next character by weighted probability
            r = random.random() * total
            upto = 0
            for c, w in zip(chars, counts):
                upto += w
                if r < upto:
                    result += c
                    break
        return result

    def visualize(self) -> str:
        """
        Return a Mermaid diagram of the Markov chain states (k-grams).
        """
        net = Network()
        # nodes are k-grams
        for key in self.model.keys():
            node_id = f"s_{key.replace(' ', '_')}"
            net.add_node(node_id, key)
        # edges are transitions
        for key, transitions in self.model.items():
            src_id = f"s_{key.replace(' ', '_')}"
            for nxt in transitions.keys():
                if self.k > 1:
                    dst_key = key[1:] + nxt
                else:
                    dst_key = nxt
                dst_id = f"s_{dst_key.replace(' ', '_')}"
                if dst_id in net._nodes:
                    net.connect(src_id, dst_id)
        return net.visualize()
