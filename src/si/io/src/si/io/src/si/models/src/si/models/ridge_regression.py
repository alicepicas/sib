import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001,
                 max_iter: int = 2000, patience: int = 5,
                 scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> "RidgeRegression":
        X, y = dataset.X, dataset.y

        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1
            X = (X - self.mean) / self.std

        m, n = X.shape
        self.theta = np.zeros(n)
        self.theta_zero = 0.0

        best_cost = np.inf
        early_stopping = 0

        for i in range(self.max_iter):
            y_pred = np.dot(X, self.theta) + self.theta_zero
            error = y_pred - y

            gradient = (self.alpha / m) * np.dot(error, X)
            penalization_term = self.theta * (1 - self.alpha * self.l2_penalty / m)

            self.theta = penalization_term - gradient
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(error)

            cost = self.cost(dataset)
            self.cost_history[i] = cost

            if cost < best_cost:
                best_cost = cost
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping >= self.patience:
                    break

        return self


    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.X
        if self.scale:
            X = (X - self.mean) / self.std
        return np.dot(X, self.theta) + self.theta_zero