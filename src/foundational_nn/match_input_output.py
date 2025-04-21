import numpy as np

from .cell import Cell

class MatchInputOutput(Cell):
    """Used in autoencoders for reconstruction."""

    def __init__(self, input_data: np.ndarray, encoder_weights: np.ndarray, decoder_weights: np.ndarray) -> None:
        super().__init__(inputs=input_data)
        self.input_data: np.ndarray = input_data
        self.encoder_weights: np.ndarray = encoder_weights
        self.decoder_weights: np.ndarray = decoder_weights

    def compute(self) -> np.ndarray:
        encoded = np.dot(self.input_data, self.encoder_weights)
        self.outputs = np.dot(encoded, self.decoder_weights)
        return self.outputs

    def visualize(self) -> str:
        return f"MatchInputOutput[{self.outputs}]"