from abc import ABCMeta, abstractmethod
 
import numpy as np
 
 
class LossFunction(metaclass=ABCMeta):
    """
    Abstract base class for the loss functions of a neural network.
    A loss function quantifies the difference between the real and the predicted values,
    and guides the optimization: the goal of the training is to minimize it.
    """
 
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the loss between the real and the predicted values.
        Abstract method that needs to be implemented by all subclasses.
 
        Parameters
        ----------
        y_true: np.ndarray
            The real values
        y_pred: np.ndarray
            The predicted values
 
        Returns
        -------
        loss: float
            The value of the loss
        """
        raise NotImplementedError
 
    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss with respect to the predicted values.
        Abstract method that needs to be implemented by all subclasses.
 
        Parameters
        ----------
        y_true: np.ndarray
            The real values
        y_pred: np.ndarray
            The predicted values
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the loss with respect to the predicted values
        """
        raise NotImplementedError
 
 
class MeanSquaredError(LossFunction):
    """
    The mean squared error loss function.
    It measures the average squared difference between the real and the predicted values,
    and is used in regression tasks.
    """
 
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the mean squared error: L = (1/n) sum (y - y*)^2.
 
        Parameters
        ----------
        y_true: np.ndarray
            The real values
        y_pred: np.ndarray
            The predicted values
 
        Returns
        -------
        loss: float
            The mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)
 
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the mean squared error: dL/dY = (2/n) (Y* - Y).
 
        Parameters
        ----------
        y_true: np.ndarray
            The real values
        y_pred: np.ndarray
            The predicted values
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the mean squared error with respect to the predicted values
        """
        return 2 * (y_pred - y_true) / y_true.size
 
 
class BinaryCrossEntropy(LossFunction):
    """
    The binary cross-entropy loss function.
    It quantifies the dissimilarity between the predicted probabilities and the real binary labels,
    and is used in binary classification tasks.
    """
 
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the binary cross-entropy: L = - sum (y log(y*) + (1 - y) log(1 - y*)).
        The predictions are clipped to avoid log(0).
 
        Parameters
        ----------
        y_true: np.ndarray
            The real binary labels
        y_pred: np.ndarray
            The predicted probabilities
 
        Returns
        -------
        loss: float
            The binary cross-entropy
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
 
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross-entropy: dL/dY = -Y / Y* + (1 - Y) / (1 - Y*).
        The predictions are clipped to avoid divisions by zero.
 
        Parameters
        ----------
        y_true: np.ndarray
            The real binary labels
        y_pred: np.ndarray
            The predicted probabilities
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the binary cross-entropy with respect to the predicted probabilities
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / p + (1 - y_true) / (1 - p)
 
 
class CategoricalCrossEntropy(LossFunction):
    """
    The categorical cross-entropy loss function.
    It measures the dissimilarity between the predicted class probabilities and the real
    one-hot encoded labels, and is used in multi-class classification tasks.
    """
 
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the categorical cross-entropy: L = - sum y log(y*).
        The predictions are clipped to avoid log(0).
 
        Parameters
        ----------
        y_true: np.ndarray
            The real one-hot encoded labels
        y_pred: np.ndarray
            The predicted class probabilities
 
        Returns
        -------
        loss: float
            The categorical cross-entropy
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p))
 
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the categorical cross-entropy: dL/dY = -Y / Y*.
        The predictions are clipped to avoid divisions by zero.
 
        Parameters
        ----------
        y_true: np.ndarray
            The real one-hot encoded labels
        y_pred: np.ndarray
            The predicted class probabilities
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the categorical cross-entropy with respect to the predicted probabilities
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / p
 