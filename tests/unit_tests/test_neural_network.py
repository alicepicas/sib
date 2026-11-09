import unittest
 
import numpy as np
 
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import MeanSquaredError, BinaryCrossEntropy
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD, Adam
 
 
def xor_dataset():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    return Dataset(X, y)
 
 
def xor_network(optimizer=SGD, learning_rate=0.5, epochs=5000, hidden_activation=True):
    network = NeuralNetwork(epochs=epochs, batch_size=4, optimizer=optimizer, learning_rate=learning_rate,
                            loss=MeanSquaredError, metric=accuracy)
    network.add(DenseLayer(4, (2,)))
    if hidden_activation:
        network.add(SigmoidActivation())
    network.add(DenseLayer(1))
    network.add(SigmoidActivation())
    return network
 
 
class TestNeuralNetwork(unittest.TestCase):
 
    def test_add_sets_shapes_and_initializes_layers(self):
        network = NeuralNetwork()
        network.add(DenseLayer(4, (2,)))
        network.add(SigmoidActivation())
        network.add(DenseLayer(1))
 
        self.assertEqual(3, len(network.layers))
        self.assertEqual((4,), network.layers[1].input_shape())
        self.assertEqual((4,), network.layers[2].input_shape())
        self.assertEqual((4, 1), network.layers[2].weights.shape)
        self.assertIsNotNone(network.layers[0].w_opt)
        self.assertIsNotNone(network.layers[0].b_opt)
 
    def test_xor(self):
        dataset = xor_dataset()
        np.random.seed(1)
        network = xor_network()
        network.fit(dataset)
 
        predictions = network.predict(dataset)
 
        self.assertEqual((4, 1), predictions.shape)
        np.testing.assert_array_equal(np.array([0, 1, 1, 0]), predictions.round().ravel())
        self.assertAlmostEqual(0.05, predictions[0, 0], delta=0.02)
        self.assertAlmostEqual(0.95, predictions[1, 0], delta=0.02)
        self.assertEqual(1.0, network.score(dataset))
 
    def test_history(self):
        dataset = xor_dataset()
        np.random.seed(1)
        network = xor_network(epochs=300)
        network.fit(dataset)
 
        self.assertEqual(300, len(network.history))
        self.assertEqual({'loss', 'metric'}, set(network.history[1].keys()))
        self.assertLess(network.history[300]['loss'], network.history[1]['loss'])
 
    def test_xor_with_adam(self):
        dataset = xor_dataset()
        np.random.seed(1)
        network = xor_network(optimizer=Adam, learning_rate=0.1, epochs=2000)
        network.fit(dataset)
 
        np.testing.assert_array_equal(np.array([0, 1, 1, 0]), network.predict(dataset).round().ravel())
 
    def test_xor_without_the_hidden_activation_is_not_learned(self):
        dataset = xor_dataset()
        np.random.seed(1)
        network = xor_network(hidden_activation=False)
        network.fit(dataset)
 
        np.testing.assert_allclose(np.full(4, 0.5), network.predict(dataset).ravel(), atol=0.05)
 
    def test_binary_classification_with_binary_cross_entropy(self):
        np.random.seed(0)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        dataset = Dataset(X, y)
 
        network = NeuralNetwork(epochs=60, batch_size=16, optimizer=SGD, learning_rate=0.05,
                                loss=BinaryCrossEntropy, metric=accuracy)
        network.add(DenseLayer(4, (5,)))
        network.add(ReLUActivation())
        network.add(DenseLayer(1))
        network.add(SigmoidActivation())
        network.fit(dataset)
 
        self.assertLess(network.history[60]['loss'], network.history[1]['loss'])
        self.assertGreater(network.score(dataset), 0.85)
 
    def test_mini_batches_cover_all_the_samples(self):
        network = NeuralNetwork(batch_size=4)
        X = np.arange(10, dtype=float).reshape(10, 1)
        y = np.arange(10, dtype=float).reshape(10, 1)
 
        batches = list(network._get_mini_batches(X, y))
 
        self.assertEqual([4, 4, 2], [len(batch[0]) for batch in batches])
        self.assertEqual(list(range(10)), sorted(np.concatenate([batch[0] for batch in batches]).ravel().tolist()))
        for X_batch, y_batch in batches:
            np.testing.assert_array_equal(X_batch, y_batch)
 
    def test_accuracy_with_one_hot_labels(self):
        network = NeuralNetwork(metric=accuracy)
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
        y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
 
        self.assertEqual(0.75, network._evaluate_metric(y_true, y_pred))
 
    def test_score_before_fit(self):
        with self.assertRaises(ValueError):
            xor_network().score(xor_dataset())