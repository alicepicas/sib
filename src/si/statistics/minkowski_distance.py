import numpy as np
 
 
def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float = 2) -> np.ndarray:
    """
    Computes the minkowski distance between a single sample and multiple samples.
    It generalizes the manhattan distance (p = 1) and the euclidean distance (p = 2).
 
    Parameters
    ----------
    x: np.ndarray
        A single sample (n_features)
    y: np.ndarray
        Multiple samples (n_samples, n_features)
    p: float
        The order of the distance (p >= 1)
 
    Returns
    -------
    distances: np.ndarray
        The minkowski distance between x and each sample of y (n_samples)
    """
    return np.sum(np.abs(x - y) ** p, axis=1) ** (1 / p)