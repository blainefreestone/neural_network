import numpy as np

class Activation:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forwards the input data through the activation function.
        The activation function is ReLU.
        """
        self.output = np.maximum(0, inputs)