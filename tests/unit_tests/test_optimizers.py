import unittest
 
import numpy as np
 
from si.neural_networks.optimizers import Optimizer, SGD, Adam
 
 
class TestOptimizers(unittest.TestCase):
 
    def test_sgd_without_momentum(self):
        sgd = SGD(learning_rate=0.1, momentum=0.0)
        w = np.array([1.0, 2.0])
        grad = np.array([0.5, -1.0])
 
        np.testing.assert_allclose(w - 0.1 * grad, sgd.update(w, grad))
 
    def test_sgd_retained_gradient(self):
        sgd = SGD(learning_rate=0.1, momentum=0.9)
        w = np.array([1.0, 2.0])
        grad = np.array([1.0, -2.0])
 
        self.assertIsNone(sgd.retained_gradient)
        sgd.update(w, grad)
        np.testing.assert_allclose(0.1 * grad, sgd.retained_gradient)
        sgd.update(w, grad)
        np.testing.assert_allclose((0.9 * 0.1 + 0.1) * grad, sgd.retained_gradient)
 
    def test_sgd_instances_do_not_share_state(self):
        first, second = SGD(), SGD()
        first.update(np.zeros(3), np.ones(3))
 
        self.assertIsNone(second.retained_gradient)
 
    def test_sgd_minimizes_a_function(self):
        sgd = SGD(learning_rate=0.5, momentum=0.0)
        w = np.array([0.0])
        for _ in range(100):
            w = sgd.update(w, 2 * (w - 3))
 
        self.assertAlmostEqual(3.0, w[0], places=3)
 
    def test_adam_first_step(self):
        adam = Adam(learning_rate=0.1)
        w = np.array([1.0, 1.0])
        grad = np.array([0.5, -4.0])
 
        new_w = adam.update(w, grad)
 
        np.testing.assert_allclose(w - 0.1 * np.sign(grad), new_w, atol=1e-6)
        self.assertEqual(1, adam.t)
 
    def test_adam_moving_averages(self):
        adam = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999)
        w = np.array([0.0])
        adam.update(w, np.array([2.0]))
        adam.update(w, np.array([4.0]))
 
        m = 0.9 * (0.1 * 2.0) + 0.1 * 4.0
        v = 0.999 * (0.001 * 4.0) + 0.001 * 16.0
        np.testing.assert_allclose([m], adam.m)
        np.testing.assert_allclose([v], adam.v)
        self.assertEqual(2, adam.t)
 
    def test_adam_minimizes_a_function(self):
        adam = Adam(learning_rate=0.1)
        w = np.array([0.0])
        for _ in range(500):
            w = adam.update(w, 2 * (w - 3))
 
        self.assertAlmostEqual(3.0, w[0], delta=0.05)
 
    def test_optimizer_is_abstract(self):
        with self.assertRaises(TypeError):
            Optimizer(0.1)