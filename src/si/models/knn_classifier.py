from typing import Callable, Any
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
 
 
class KNNClassifier(Model):
    """
    The k-Nearest Neighbors classifier.
    It estimates the class of a sample based on the k most similar samples of the training dataset.
 
    Parameters
    ----------
    k: int
        The number of nearest samples to consider
    distance: Callable
        A function that calculates the distance between a sample and the samples of the training dataset.
        It receives a single sample (n_features) and the training samples (n_samples, n_features),
        and returns the distances (n_samples). The euclidean distance is used by default
 
    Attributes
    ----------
    dataset: Dataset
        The training dataset
    """
 
    def __init__(self, k: int = 3, distance: Callable = euclidean_distance):
        """
        Initialize the KNN classifier.
 
        Parameters
        ----------
        k: int
            The number of nearest samples to consider
        distance: Callable
            A function that calculates the distance between a sample and the samples of the training dataset
        """
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset = None
 
    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Stores the training dataset.
 
        Parameters
        ----------
        dataset: Dataset
            The training dataset
 
        Returns
        -------
        self: KNNClassifier
            The fitted model
        """
        self.dataset = dataset
        return self
 
    def _predict_sample(self, sample: np.ndarray) -> Any:
        """
        Estimates the class of a single sample.
        It is the most common class among the k most similar samples of the training dataset.
 
        Parameters
        ----------
        sample: np.ndarray
            The sample to classify (n_features)
 
        Returns
        -------
        label: Any
            The predicted class
        """
        distances = self.distance(sample, self.dataset.X)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.dataset.y[k_nearest_indices]
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]
 
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the class of each sample of the dataset.
 
        Parameters
        ----------
        dataset: Dataset
            The testing dataset
 
        Returns
        -------
        predictions: np.ndarray
            The predicted classes (n_samples)
        """
        return np.array([self._predict_sample(sample) for sample in dataset.X])
 
    def _score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy between the predicted and the real classes.
 
        Parameters
        ----------
        dataset: Dataset
            The testing dataset
 
        Returns
        -------
        accuracy: float
            The proportion of correctly classified samples
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)