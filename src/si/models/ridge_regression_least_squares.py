import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse
 
 
class RidgeRegressionLeastSquares(Model):
    """
    The RidgeRegressionLeastSquares is a linear model using the L2 regularization.
    This model solves the linear regression problem using the closed form of the Least Squares method.
 
    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
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
    """
 
    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        """
        Initialize the RidgeRegressionLeastSquares model.
 
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        scale: bool
            Whether to scale the dataset or not
        """
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale
 
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
 
    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estimates theta and theta_zero (and the mean and std, if the data is scaled) using the Least Squares method.
        The parameters are given by (X^T X + l2_penalty * I)^-1 X^T y, where the intercept is not penalized.
 
        Parameters
        ----------
        dataset: Dataset
            The training dataset
 
        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """
        X, y = dataset.X, dataset.y
 
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1
            X = (X - self.mean) / self.std
 
        X = np.c_[np.ones(X.shape[0]), X]
 
        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])
        penalty_matrix[0, 0] = 0
 
        thetas = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(y)
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]
 
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
 
        X = np.c_[np.ones(X.shape[0]), X]
        thetas = np.r_[self.theta_zero, self.theta]
 
        return X.dot(thetas)
 
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