import numpy as np
 
 
def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function, which maps any real value to the interval (0, 1).
 
    Parameters
    ----------
    x: np.ndarray
        The input values
 
    Returns
    -------
    sigmoid: np.ndarray
        The value of the sigmoid function for each input value
    """
    return 1 / (1 + np.exp(-x))

 