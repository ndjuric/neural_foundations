from foundational_nn.hidden_cell import HiddenCell
from architecture_nn.network import Network


class HopfieldNetwork:
    """
    Hopfield network for associative memory using atomic primitives.
    """

    def __init__(self, n_units: int):
        self.n_units = n_units
        # symmetric weight matrix
        self.weights = [[0.0] * n_units for _ in range(n_units)]

    def train(self, patterns):
        """
        Train the network with Hebbian rule.
        patterns: iterable of patterns, each a list of ±1 values.
        """
        for p in patterns:
            for i in range(self.n_units):
                for j in range(self.n_units):
                    self.weights[i][j] += p[i] * p[j]
        # zero diagonal
        for i in range(self.n_units):
            self.weights[i][i] = 0.0

    def recall(self, state, steps: int = 5):
        """
        Recall a stored pattern from an initial state.
        state: list of ±1 values.
        """
        s = list(state)

        # simple sign activation
        def sign_activation(x):
            if hasattr(x, "__iter__"):
                return [1 if xi >= 0 else -1 for xi in x]
            return 1 if x >= 0 else -1

        for _ in range(steps):
            for i in range(self.n_units):
                cell = HiddenCell(
                    inputs=s,
                    weights=self.weights[i],
                    bias=0.0,
                    activation=sign_activation,
                )
                out = cell.compute()
                # out may be list or scalar
                s[i] = out[0] if hasattr(out, "__iter__") else out
        return s

    def visualize(self) -> str:
        """
        Return a Mermaid diagram of the fully-connected Hopfield network.
        """
        net = Network()
        for i in range(self.n_units):
            net.add_node(f"u{i}", f"U{i}")
        for i in range(self.n_units):
            for j in range(i + 1, self.n_units):
                net.connect(f"u{i}", f"u{j}")
                net.connect(f"u{j}", f"u{i}")
        return net.visualize()
