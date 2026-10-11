import numpy as np
 
 
def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the manhattan distance between a single sample and multiple samples.
    Each distance is the sum of the absolute differences between the features.
 
    Parameters
    ----------
    x: np.ndarray
        A single sample (n_features)
    y: np.ndarray
        Multiple samples (n_samples, n_features)
 
    Returns
    -------
    distances: np.ndarray
        The manhattan distance between x and each sample of y (n_samples)
    """
    return np.sum(np.abs(x - y), axis=1)