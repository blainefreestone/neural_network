import unittest
import numpy as np
from layer import Layer

class TestLayer(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.layer = Layer(3, 2)

    def test_init(self):
        self.assertEqual(self.layer.weights.shape, (3, 2))
        self.assertEqual(self.layer.biases.shape, (1, 2))
        self.assertTrue(np.all(self.layer.biases == np.zeros((1, 2))))

    def test_forward(self):
        inputs = np.array([[1, 2, 3]])
        expected_output = np.dot(inputs, self.layer.weights) + self.layer.biases
        np.testing.assert_array_equal(self.layer.forward(inputs), expected_output)

if __name__ == '__main__':
    unittest.main()