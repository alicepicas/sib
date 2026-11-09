from typing import Callable
 
import numpy as np
 
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse
from si.neural_networks.losses import LossFunction, MeanSquaredError
from si.neural_networks.optimizers import Optimizer, SGD
 
 
class NeuralNetwork(Model):
    """
    The NeuralNetwork connects the layers, the loss function, the optimizer and the metric,
    and trains the network with mini-batches: forward propagation, loss derivative,
    backward propagation and update of the weights, repeated for many epochs.
 
    Parameters
    ----------
    epochs: int
        The number of epochs to train the neural network
    batch_size: int
        The batch size to use for training the neural network
    optimizer: Optimizer
        The optimizer class to use for training the neural network
    learning_rate: float
        The learning rate to use for training the neural network
    verbose: bool
        Whether to print the loss and the metric at each epoch
    loss: LossFunction
        The loss function class to use for training the neural network
    metric: Callable
        The metric to use for training the neural network
    kwargs
        Additional keyword arguments passed to the optimizer
 
    Attributes
    ----------
    layers: list
        The layers of the neural network
    history: dict
        The loss and the metric at each epoch
    """
 
    def __init__(self, epochs: int = 100, batch_size: int = 128, optimizer: type = SGD,
                 learning_rate: float = 0.01, verbose: bool = False, loss: type = MeanSquaredError,
                 metric: Callable = mse, **kwargs):
        """
        Initialize the neural network.
 
        Parameters
        ----------
        epochs: int
            The number of epochs to train the neural network
        batch_size: int
            The batch size to use for training the neural network
        optimizer: Optimizer
            The optimizer class to use for training the neural network
        learning_rate: float
            The learning rate to use for training the neural network
        verbose: bool
            Whether to print the loss and the metric at each epoch
        loss: LossFunction
            The loss function class to use for training the neural network
        metric: Callable
            The metric to use for training the neural network
        kwargs
            Additional keyword arguments passed to the optimizer
        """
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
 
        self.layers = []
        self.history = {}
 
    def add(self, layer) -> 'NeuralNetwork':
        """
        Adds a layer to the neural network.
        The input shape of the layer is set from the output shape of the previous layer
        (the first layer must be given its input shape), and the layer is initialized
        with the optimizer, if it needs one.
 
        Parameters
        ----------
        layer
            The layer to add to the neural network
 
        Returns
        -------
        self: NeuralNetwork
            The neural network
        """
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
 
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
 
        self.layers.append(layer)
        return self
 
    @staticmethod
    def _as_column(y: np.ndarray) -> np.ndarray:
        """
        Converts a vector of labels into a column matrix, the shape of the output of the network.
 
        Parameters
        ----------
        y: np.ndarray
            The labels
 
        Returns
        -------
        y: np.ndarray
            The labels as a matrix with one row per sample
        """
        return y.reshape(-1, 1) if y.ndim == 1 else y
 
    def _get_mini_batches(self, X: np.ndarray, y: np.ndarray):
        """
        Shuffles the samples and splits them into mini-batches of batch_size samples
        (the last one can be smaller).
 
        Parameters
        ----------
        X: np.ndarray
            The input data
        y: np.ndarray
            The labels
 
        Returns
        -------
        batches: generator
            The mini-batches, as pairs (X_batch, y_batch)
        """
        indices = np.random.permutation(X.shape[0])
        for start in range(0, X.shape[0], self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield X[batch], y[batch]
 
    def _evaluate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the metric. For the accuracy, the predicted probabilities are first converted
        into classes (a threshold of 0.5 for a single output, or the highest probability for several outputs).
 
        Parameters
        ----------
        y_true: np.ndarray
            The real labels
        y_pred: np.ndarray
            The predictions of the network
 
        Returns
        -------
        metric: float
            The value of the metric
        """
        if self.metric is accuracy:
            if y_pred.shape[1] == 1:
                return accuracy(y_true.ravel(), np.where(y_pred >= 0.5, 1, 0).ravel())
            return accuracy(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        return self.metric(y_true, y_pred)
 
    def _fit(self, dataset: Dataset) -> 'NeuralNetwork':
        """
        Trains the neural network. For each epoch, it shuffles the data, splits it into mini-batches
        and, for each mini-batch, does the forward propagation through all the layers, computes the loss
        derivative, and does the backward propagation, which updates the weights. At the end of each epoch,
        it computes the loss and the metric and stores them in the history.
 
        Parameters
        ----------
        dataset: Dataset
            The dataset to train the neural network on
 
        Returns
        -------
        self: NeuralNetwork
            The trained neural network
        """
        X = dataset.X
        y = self._as_column(dataset.y)
        self.history = {}
 
        for epoch in range(1, self.epochs + 1):
            outputs = []
            labels = []
 
            for X_batch, y_batch in self._get_mini_batches(X, y):
                output = X_batch
                for layer in self.layers:
                    output = layer.forward_propagation(output, training=True)
 
                error = self.loss.derivative(y_batch, output)
                for layer in self.layers[::-1]:
                    error = layer.backward_propagation(error)
 
                outputs.append(output)
                labels.append(y_batch)
 
            outputs = np.concatenate(outputs)
            labels = np.concatenate(labels)
 
            loss = float(self.loss.loss(labels, outputs))
            metric = float(self._evaluate_metric(labels, outputs)) if self.metric is not None else 'NA'
            self.history[epoch] = {'loss': loss, 'metric': metric}
 
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - metric: {metric}')
 
        return self
 
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on new data, with the forward propagation through all the layers
        (not in training mode).
 
        Parameters
        ----------
        dataset: Dataset
            The dataset to make predictions on
 
        Returns
        -------
        predictions: np.ndarray
            The output of the last layer for each sample
        """
        output = dataset.X
        for layer in self.layers:
            output = layer.forward_propagation(output, training=False)
        return output
 
    def _score(self, dataset: Dataset) -> float:
        """
        Evaluates the neural network on a dataset, computing the metric between the real labels
        and the predictions.
 
        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the neural network on
 
        Returns
        -------
        metric: float
            The value of the metric
        """
        predictions = self._predict(dataset)
        return self._evaluate_metric(self._as_column(dataset.y), predictions)