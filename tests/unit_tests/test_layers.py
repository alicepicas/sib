import unittest
import numpy as np
from si.neural_networks.layers import Layer, DenseLayer, Dropout
 
 
class NoUpdate:
 
    def update(self, w, grad):
        self.grad = grad
        return w
 
 
class GradientDescent:
 
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
 
    def update(self, w, grad):
        return w - self.learning_rate * grad
 
 
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
 
    def test_backward_propagation_input_error(self):
        np.random.seed(0)
        layer = DenseLayer(2, (3,)).initialize(NoUpdate())
        layer.forward_propagation(np.random.rand(4, 3), training=True)
        output_error = np.random.rand(4, 2)
 
        input_error = layer.backward_propagation(output_error)
 
        self.assertEqual((4, 3), input_error.shape)
        np.testing.assert_allclose(np.dot(output_error, layer.weights.T), input_error)
 
    def test_backward_propagation_errors_of_weights_and_biases(self):
        np.random.seed(0)
        X = np.random.rand(4, 3)
        output_error = np.random.rand(4, 2)
        layer = DenseLayer(2, (3,)).initialize(NoUpdate())
        layer.forward_propagation(X, training=True)
 
        layer.backward_propagation(output_error)
 
        np.testing.assert_allclose(np.dot(X.T, output_error), layer.w_opt.grad)
        np.testing.assert_allclose(np.sum(output_error, axis=0, keepdims=True), layer.b_opt.grad)
        self.assertEqual((1, 2), layer.b_opt.grad.shape)
 
    def test_backward_propagation_gradient_check(self):
        np.random.seed(0)
        X, Y = np.random.rand(4, 3), np.random.rand(4, 2)
        mse = lambda Y, P: np.mean((Y - P) ** 2)
        layer = DenseLayer(2, (3,)).initialize(NoUpdate())
        output = layer.forward_propagation(X, training=True)
        layer.backward_propagation(2 * (output - Y) / Y.size)
        analytic = layer.w_opt.grad
 
        numeric = np.zeros_like(layer.weights)
        eps = 1e-6
        for i in range(3):
            for j in range(2):
                layer.weights[i, j] += eps
                plus = mse(Y, layer.forward_propagation(X, True))
                layer.weights[i, j] -= 2 * eps
                minus = mse(Y, layer.forward_propagation(X, True))
                layer.weights[i, j] += eps
                numeric[i, j] = (plus - minus) / (2 * eps)
 
        self.assertLess(np.abs(numeric - analytic).max(), 1e-7)
 
    def test_backward_propagation_updates_weights_and_biases(self):
        np.random.seed(0)
        X = np.random.rand(4, 3)
        output_error = np.random.rand(4, 2)
        layer = DenseLayer(2, (3,)).initialize(GradientDescent(0.1))
        layer.forward_propagation(X, training=True)
        weights, biases = layer.weights.copy(), layer.biases.copy()
 
        layer.backward_propagation(output_error)
 
        np.testing.assert_allclose(weights - 0.1 * np.dot(X.T, output_error), layer.weights)
        np.testing.assert_allclose(biases - 0.1 * np.sum(output_error, axis=0, keepdims=True), layer.biases)
 
 
class TestDropout(unittest.TestCase):
 
    def test_forward_propagation_training(self):
        np.random.seed(0)
        layer = Dropout(0.5)
        X = np.ones((1000, 10))
 
        output = layer.forward_propagation(X, training=True)
 
        self.assertEqual(X.shape, output.shape)
        self.assertAlmostEqual(0.5, (output == 0).mean(), delta=0.05)
        self.assertEqual([0.0, 2.0], np.unique(output).tolist())
 
    def test_forward_propagation_inference(self):
        layer = Dropout(0.5)
        X = np.random.rand(10, 4)
 
        output = layer.forward_propagation(X, training=False)
 
        np.testing.assert_array_equal(X, output)
 
    def test_backward_propagation(self):
        np.random.seed(0)
        layer = Dropout(0.3)
        X = np.random.rand(20, 5)
        layer.forward_propagation(X, training=True)
        output_error = np.random.rand(20, 5)
 
        input_error = layer.backward_propagation(output_error)
 
        np.testing.assert_array_equal(output_error * layer.mask, input_error)
 
    def test_output_shape_and_parameters(self):
        layer = Dropout(0.2)
        layer.set_input_shape((5,))
 
        self.assertEqual((5,), layer.output_shape())
        self.assertEqual(0, layer.parameters())
 
    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            Dropout(1.0)
        with self.assertRaises(ValueError):
            Dropout(-0.1)
 