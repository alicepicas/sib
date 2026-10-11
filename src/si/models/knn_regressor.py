from typing import Callable
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance
 
 
class KNNRegressor(Model):
    """
    The k-Nearest Neighbors regressor.
    It estimates the value of a sample as the average value of the k most similar samples of the training dataset.
 
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
        Initialize the KNN regressor.
 
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
 
    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset.
 
        Parameters
        ----------
        dataset: Dataset
            The training dataset
 
        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self
 
    def _predict_sample(self, sample: np.ndarray) -> float:
        """
        Estimates the value of a single sample.
        It is the average value of the k most similar samples of the training dataset.
 
        Parameters
        ----------
        sample: np.ndarray
            The sample to estimate (n_features)
 
        Returns
        -------
        value: float
            The predicted value
        """
        distances = self.distance(sample, self.dataset.X)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_values = self.dataset.y[k_nearest_indices]
        return np.mean(k_nearest_values)
 
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the value of each sample of the dataset.
 
        Parameters
        ----------
        dataset: Dataset
            The testing dataset
 
        Returns
        -------
        predictions: np.ndarray
            The predicted values (n_samples)
        """
        return np.array([self._predict_sample(sample) for sample in dataset.X])
 
    def _score(self, dataset: Dataset) -> float:
        """
        Calculates the RMSE between the predicted and the real values.
 
        Parameters
        ----------
        dataset: Dataset
            The testing dataset
 
        Returns
        -------
        rmse: float
            The error between the predicted and the real values
        """
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)