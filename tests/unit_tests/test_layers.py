import unittest
 
import numpy as np
 
from si.neural_networks.layers import Layer, DenseLayer
 
 
class TestDenseLayer(unittest.TestCase):
 
    def test_initialize(self):
        layer = DenseLayer(4, (3,)).initialize()
 
        self.assertEqual((3, 4), layer.weights.shape)
        self.assertEqual((1, 4), layer.biases.shape)
        self.assertTrue(np.all(layer.weights >= -0.5) and np.all(layer.weights < 0.5))
        self.assertTrue(np.all(layer.biases == 0))
 
    def test_forward_propagation(self):
        layer = DenseLayer(4, (3,)).initialize()
        output = layer.forward_propagation(np.random.rand(10, 3))
 
        self.assertEqual((10, 4), output.shape)
 
    def test_forward_propagation_values(self):
        layer = DenseLayer(3, (2,)).initialize()
        layer.weights = np.array([[0.1, 0.2, -0.1], [0.3, -0.2, 0.4]])
        layer.biases = np.array([[0.1, 0, -0.1]])
 
        output = layer.forward_propagation(np.array([[1, 2]]))
 
        np.testing.assert_allclose(np.array([[0.8, -0.2, 0.6]]), output)
 
    def test_parameters(self):
        layer = DenseLayer(4, (3,)).initialize()
 
        self.assertEqual(3 * 4 + 4, layer.parameters())
 
    def test_output_shape(self):
        layer = DenseLayer(4, (3,)).initialize()
 
        self.assertEqual((4,), layer.output_shape())
 
    def test_layer_name_and_input_shape(self):
        layer = DenseLayer(4, (3,))
 
        self.assertEqual('DenseLayer', layer.layer_name())
        self.assertEqual((3,), layer.input_shape())
 
    def test_layer_is_abstract(self):
        with self.assertRaises(TypeError):
            Layer()
            