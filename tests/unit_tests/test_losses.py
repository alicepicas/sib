import unittest
 
import numpy as np
 
from si.neural_networks.losses import LossFunction, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
 
 
def numeric_derivative(loss, y_true, y_pred, eps=1e-6):
    numeric = np.zeros_like(y_pred)
    for index in np.ndindex(*y_pred.shape):
        plus, minus = y_pred.copy(), y_pred.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric[index] = (loss.loss(y_true, plus) - loss.loss(y_true, minus)) / (2 * eps)
    return numeric
 
 
class TestLosses(unittest.TestCase):
 
    def test_mean_squared_error(self):
        mse = MeanSquaredError()
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.0], [4.0]])
 
        self.assertAlmostEqual(2.0, mse.loss(y_true, y_pred))
        np.testing.assert_allclose(np.array([[0.0], [2.0]]), mse.derivative(y_true, y_pred))
 
    def test_binary_cross_entropy(self):
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.8], [0.3]])
 
        self.assertAlmostEqual(-np.log(0.8) - np.log(0.7), bce.loss(y_true, y_pred))
        np.testing.assert_allclose(np.array([[-1 / 0.8], [1 / 0.7]]), bce.derivative(y_true, y_pred))
 
    def test_categorical_cross_entropy(self):
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1.0, 0.0, 0.0]])
        y_pred = np.array([[0.7, 0.2, 0.1]])
 
        self.assertAlmostEqual(-np.log(0.7), cce.loss(y_true, y_pred))
        np.testing.assert_allclose(np.array([[-1 / 0.7, 0.0, 0.0]]), cce.derivative(y_true, y_pred))
 
    def test_derivatives_with_numeric_gradient(self):
        np.random.seed(0)
        y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
        y_pred = np.random.uniform(0.1, 0.9, (4, 1))
        one_hot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        probabilities = np.random.uniform(0.1, 0.9, (2, 3))
 
        for loss, true, pred in ((MeanSquaredError(), y_true, y_pred),
                                 (BinaryCrossEntropy(), y_true, y_pred),
                                 (CategoricalCrossEntropy(), one_hot, probabilities)):
            analytic = loss.derivative(true, pred)
            numeric = numeric_derivative(loss, true, pred)
            self.assertLess(np.abs(analytic - numeric).max(), 1e-5)
 
    def test_cross_entropies_are_finite_at_the_extremes(self):
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.0], [1.0]])
 
        for loss in (BinaryCrossEntropy(), CategoricalCrossEntropy()):
            self.assertTrue(np.isfinite(loss.loss(y_true, y_pred)))
            self.assertTrue(np.all(np.isfinite(loss.derivative(y_true, y_pred))))
 
    def test_loss_function_is_abstract(self):
        with self.assertRaises(TypeError):
            LossFunction()