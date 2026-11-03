import unittest
import numpy as np
from si.neural_networks.activation import (ActivationLayer, ReLUActivation, LeakyReLUActivation, SigmoidActivation, TanhActivation, SoftmaxActivation)
 
 
class TestActivation(unittest.TestCase):
 
    def setUp(self):
        self.x = np.array([-2.5, 1.2, 0.3, 0.4, 1.1, 2.7])
 
    def test_relu(self):
        relu = ReLUActivation()
        x = np.array([-1.0, 0.0, 2.0])
 
        np.testing.assert_array_equal(np.array([0.0, 0.0, 2.0]), relu.activation_function(x))
        np.testing.assert_array_equal(np.array([0, 1, 1]), relu.derivative(x))
 
    def test_sigmoid(self):
        sigmoid = SigmoidActivation()
 
        np.testing.assert_allclose(np.array([0.5]), sigmoid.activation_function(np.array([0.0])))
        np.testing.assert_allclose(np.array([0.25]), sigmoid.derivative(np.array([0.0])))
 
    def test_tanh(self):
        tanh = TanhActivation()
 
        np.testing.assert_allclose(np.array([0.0]), tanh.activation_function(np.array([0.0])))
        np.testing.assert_allclose(np.array([1.0]), tanh.derivative(np.array([0.0])))
        self.assertTrue(np.all(np.abs(tanh.activation_function(self.x)) < 1))
 
    def test_softmax(self):
        softmax = SoftmaxActivation()
        x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
 
        output = softmax.activation_function(x)
 
        np.testing.assert_allclose(np.ones(2), output.sum(axis=1))
        self.assertFalse(np.isnan(output).any())
        np.testing.assert_allclose(output[0], output[1])
 
    def test_derivatives_with_numeric_gradient(self):
        eps = 1e-6
        for activation in (SigmoidActivation(), ReLUActivation(), TanhActivation()):
            numeric = (activation.activation_function(self.x + eps)
                       - activation.activation_function(self.x - eps)) / (2 * eps)
            self.assertLess(np.abs(numeric - activation.derivative(self.x)).max(), 1e-7)
 
    def test_forward_and_backward_propagation(self):
        relu = ReLUActivation()
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        error = np.array([[10.0, 20.0], [30.0, 40.0]])
 
        output = relu.forward_propagation(x, training=True)
        input_error = relu.backward_propagation(error)
 
        np.testing.assert_array_equal(np.array([[0.0, 2.0], [3.0, 0.0]]), output)
        np.testing.assert_array_equal(np.array([[0.0, 20.0], [30.0, 0.0]]), input_error)
 
    def test_parameters_and_output_shape(self):
        sigmoid = SigmoidActivation()
        sigmoid.set_input_shape((4,))
 
        self.assertEqual(0, sigmoid.parameters())
        self.assertEqual((4,), sigmoid.output_shape())
 
    def test_activation_layer_is_abstract(self):
        with self.assertRaises(TypeError):
            ActivationLayer()
 
    def test_leaky_relu(self):
        leaky_relu = LeakyReLUActivation(alpha=0.1)
        x = np.array([-2.0, 0.0, 3.0])
 
        np.testing.assert_allclose(np.array([-0.2, 0.0, 3.0]), leaky_relu.activation_function(x))
        np.testing.assert_allclose(np.array([0.1, 1.0, 1.0]), leaky_relu.derivative(x))
 
    def test_leaky_relu_default_alpha(self):
        self.assertEqual(0.01, LeakyReLUActivation().alpha)
 
    def test_leaky_relu_derivative_with_numeric_gradient(self):
        leaky_relu = LeakyReLUActivation(alpha=0.2)
        eps = 1e-6
        numeric = (leaky_relu.activation_function(self.x + eps)
                   - leaky_relu.activation_function(self.x - eps)) / (2 * eps)
 
        self.assertLess(np.abs(numeric - leaky_relu.derivative(self.x)).max(), 1e-7)
 
    def test_leaky_relu_forward_and_backward_propagation(self):
        leaky_relu = LeakyReLUActivation(alpha=0.5)
        x = np.array([[-2.0, 4.0]])
        error = np.array([[10.0, 10.0]])
 
        output = leaky_relu.forward_propagation(x, training=True)
        input_error = leaky_relu.backward_propagation(error)
 
        np.testing.assert_allclose(np.array([[-1.0, 4.0]]), output)
        np.testing.assert_allclose(np.array([[5.0, 10.0]]), input_error)