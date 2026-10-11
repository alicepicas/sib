import numpy as np
 
 
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the accuracy of the model on the dataset.
 
    Parameters
    ----------
    y_true: np.ndarray
        The real label values
    y_pred: np.ndarray
        The predicted label values
 
    Returns
    -------
    accuracy: float
        The proportion of correctly classified samples
    """
    return np.sum(y_true == y_pred) / len(y_true)
