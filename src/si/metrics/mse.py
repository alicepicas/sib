import numpy as np
 
 
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the mean squared error (MSE) between the real and the predicted values.
 
    Parameters
    ----------
    y_true: np.ndarray
        The real values of y
    y_pred: np.ndarray
        The predicted values of y
 
    Returns
    -------
    mse: float
        The error between y_true and y_pred
    """
    return np.sum((y_true - y_pred) ** 2) / len(y_true)
 