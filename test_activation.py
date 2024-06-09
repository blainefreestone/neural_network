import unittest
import numpy as np
from activation import Activation

class TestActivation(unittest.TestCase):
    def setUp(self):
        self.activation = Activation()

    def test_forward(self):
        inputs = np.array([[-1, 2], [0, -3], [4, -5]])
        expected_output = np.array([[0, 2], [0, 0], [4, 0]])
        self.activation.forward(inputs)
        np.testing.assert_array_equal(self.activation.output, expected_output)

if __name__ == '__main__':
    unittest.main()