import numpy as np

class Layer:
    """
    Class for internal layers of the network
    """
    def __init__(self, num_inputs: int, num_neurons: int) -> None:
        """
        Initialize the layer with the input and output sizes.

        Args:
            num_inputs: Number of input nodes
            num_neurons: Number of output nodes
        Returns:
            None
        """
        np.random.seed(0)   # seed for reproducibility
        self.weights = np.random.randn(num_inputs, num_neurons) # start layer with random weights
        self.biases = np.zeros((1, num_neurons))    # start layer with zero biases

    def forward():
        pass