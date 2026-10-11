import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse
 
 
class RidgeRegression(Model):
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique.
 
    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations
    patience: int
        The maximum number of iterations without improvement allowed
    scale: bool
        Whether to scale the dataset or not
 
    Attributes
    ----------
    theta: np.ndarray
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    mean: np.ndarray
        The mean of each feature, used to scale the dataset
    std: np.ndarray
        The standard deviation of each feature, used to scale the dataset
    cost_history: dict
        The value of the cost function at each iteration (iteration: cost)
    """
 
    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001,
                 max_iter: int = 2000, patience: int = 5,
                 scale: bool = True, **kwargs):
        """
        Initialize the RidgeRegression model.
 
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        patience: int
            The maximum number of iterations without improvement allowed
        scale: bool
            Whether to scale the dataset or not
        """
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
        """
        Estimates theta and theta_zero (and the mean and std, if the data is scaled) using gradient descent.
        The training stops after max_iter iterations or when the cost does not improve for patience iterations.
 
        Parameters
        ----------
        dataset: Dataset
            The training dataset
 
        Returns
        -------
        self: RidgeRegression
            The fitted model
        """
        X, y = dataset.X, dataset.y
 
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1
            X = (X - self.mean) / self.std
 
        m, n = X.shape
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}
 
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
        """
        Predicts the dependent variable (y) using the estimated theta and theta_zero.
 
        Parameters
        ----------
        dataset: Dataset
            The testing dataset
 
        Returns
        -------
        predictions: np.ndarray
            The predicted values (n_samples)
        """
        X = dataset.X
        if self.scale:
            X = (X - self.mean) / self.std
        return np.dot(X, self.theta) + self.theta_zero
 
    def _score(self, dataset: Dataset) -> float:
        """
        Calculates the MSE between the predicted and the real values.
 
        Parameters
        ----------
        dataset: Dataset
            The testing dataset
 
        Returns
        -------
        mse: float
            The error between the predicted and the real values
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)
 
    def cost(self, dataset: Dataset) -> float:
        """
        Calculates the cost function (with the L2 regularization term) between the predicted and the real values.
 
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on
 
        Returns
        -------
        cost: float
            The value of the cost function
        """
        y_pred = self._predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + self.l2_penalty * np.sum(self.theta ** 2)) / (2 * len(dataset.y))
    