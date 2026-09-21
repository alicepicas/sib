import numpy as np
from typing import Callable
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse

class KNNRegressor(Model):
    def __init__(self, k: int = 3, distance: Callable = None):
        super().__init__()
        self.k = k
        self.distance = distance if distance else self._euclidean_distance
        self.dataset = None

    @staticmethod
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x1 - x2, axis=1)

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        self.dataset = dataset
        return self

    def _predict_sample(self, sample: np.ndarray) -> float:
        distances = self.distance(self.dataset.X, sample)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_values = self.dataset.y[k_nearest_indices]
        return np.mean(k_nearest_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        return np.array([self._predict_sample(sample) for sample in dataset.X])

    def _score(self, dataset: Dataset) -> float:
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)