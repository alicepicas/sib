from typing import Tuple
import numpy as np
from scipy.stats import f_oneway
from si.data.dataset import Dataset
 
 
def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scores the features of a Dataset object with the ANOVA F-test (one-way analysis of variance).
    The samples are grouped by class, and each feature is tested to see
    whether its mean differs significantly between the classes.
 
    Parameters
    ----------
    dataset: Dataset
        A labeled Dataset object
 
    Returns
    -------
    F: np.ndarray
        The F value of each feature (n_features)
    p: np.ndarray
        The p value of each feature (n_features)
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = f_oneway(*groups)
    return F, p