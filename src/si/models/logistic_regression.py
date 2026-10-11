import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function
 
 
class LogisticRegression(Model):
    """
    The LogisticRegression is a linear model for binary classification, using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique.
    The labels of the dataset must be 0 or 1.
    Optionally, the gradient descent can use momentum and the cost can have an L1 regularization term.
 
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
    momentum: float
        The momentum of the gradient descent, between 0 (no momentum) and 1 (exclusive)
    l1_penalty: float
        The L1 regularization parameter (0 for no L1 regularization)
 
    Attributes
    ----------
    theta: np.ndarray
        The model parameters, namely the coefficients of the logistic model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the logistic model.
        For example, theta_zero * 1
    mean: np.ndarray
        The mean of each feature, used to scale the dataset
    std: np.ndarray
        The standard deviation of each feature, used to scale the dataset
    cost_history: dict
        The value of the cost function at each iteration (iteration: cost)
    """
 
    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001, max_iter: int = 1000,
                 patience: int = 5, scale: bool = True, momentum: float = 0.0,
                 l1_penalty: float = 0.0, **kwargs):
        """
        Initialize the LogisticRegression model.
 
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
        momentum: float
            The momentum of the gradient descent, between 0 (no momentum) and 1 (exclusive)
        l1_penalty: float
            The L1 regularization parameter (0 for no L1 regularization)
        """
        super().__init__(**kwargs)
        if not 0 <= momentum < 1:
            raise ValueError("The momentum must be greater than or equal to 0 and lower than 1")
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        self.momentum = momentum
        self.l1_penalty = l1_penalty
 
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}
 
    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Estimates theta and theta_zero (and the mean and std, if the data is scaled) using gradient descent.
        The training stops after max_iter iterations or when the cost does not improve for patience iterations.
 
        Parameters
        ----------
        dataset: Dataset
            The training dataset
 
        Returns
        -------
        self: LogisticRegression
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
 
        velocity = np.zeros(n)
        velocity_zero = 0.0
 
        best_cost = np.inf
        early_stopping = 0
 
        for i in range(self.max_iter):
            y_pred = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
            error = y_pred - y
 
            gradient = (np.dot(error, X) / m
                        + (self.l2_penalty / m) * self.theta
                        + (self.l1_penalty / m) * np.sign(self.theta))
            gradient_zero = np.sum(error) / m
 
            velocity = self.momentum * velocity + gradient
            velocity_zero = self.momentum * velocity_zero + gradient_zero
 
            self.theta = self.theta - self.alpha * velocity
            self.theta_zero = self.theta_zero - self.alpha * velocity_zero
 
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
 
    def _probabilities(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the probability of each sample belonging to the class 1.
 
        Parameters
        ----------
        dataset: Dataset
            The dataset to estimate the probabilities of
 
        Returns
        -------
        probabilities: np.ndarray
            The estimated probabilities (n_samples)
        """
        X = dataset.X
        if self.scale:
            X = (X - self.mean) / self.std
        return sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
 
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class of each sample: 1 if the estimated probability is at least 0.5, otherwise 0.
 
        Parameters
        ----------
        dataset: Dataset
            The testing dataset
 
        Returns
        -------
        predictions: np.ndarray
            The predicted classes (n_samples)
        """
        return np.where(self._probabilities(dataset) >= 0.5, 1, 0)
 
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
 
    def cost(self, dataset: Dataset) -> float:
        """
        Calculates the cost function between the estimated probabilities and the real classes.
        It is the logistic loss plus the L2 and L1 regularization terms.
 
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on
 
        Returns
        -------
        cost: float
            The value of the cost function
        """
        m = len(dataset.y)
        probabilities = np.clip(self._probabilities(dataset), 1e-15, 1 - 1e-15)
 
        loss = -np.sum(dataset.y * np.log(probabilities) + (1 - dataset.y) * np.log(1 - probabilities)) / m
        l2_term = self.l2_penalty * np.sum(self.theta ** 2) / (2 * m)
        l1_term = self.l1_penalty * np.sum(np.abs(self.theta)) / m
 
        return loss + l2_term + l1_term
    