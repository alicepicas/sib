import numpy as np
 
 
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the euclidean distance between a single sample and multiple samples.
    Each distance is the square root of the sum of the squared differences between the features.
 
    Parameters
    ----------
    x: np.ndarray
        A single sample (n_features)
    y: np.ndarray
        Multiple samples (n_samples, n_features)
 
    Returns
    -------
    distances: np.ndarray
        The euclidean distance between x and each sample of y (n_samples)
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))
 