import numpy as np
 
 
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the root mean squared error (RMSE) between the real and the predicted values.
 
    Parameters
    ----------
    y_true: np.ndarray
        The real values of y
    y_pred: np.ndarray
        The predicted values of y
 
    Returns
    -------
    rmse: float
        The error between y_true and y_pred
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))