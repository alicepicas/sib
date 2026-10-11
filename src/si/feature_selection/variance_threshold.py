import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
 
 
class VarianceThreshold(Transformer):
    """
    Selects the features whose variance is greater than a threshold.
    Features with a low variance barely change between samples, so they carry little information.
 
    Parameters
    ----------
    threshold: float
        The cutoff value. Features with a variance lower than or equal to it are removed
 
    Attributes
    ----------
    variance: np.ndarray
        The variance of each feature, estimated from the dataset
    """
 
    def __init__(self, threshold: float = 0.0):
        """
        Initialize the VarianceThreshold transformer.
 
        Parameters
        ----------
        threshold: float
            The cutoff value. Features with a variance lower than or equal to it are removed
        """
        super().__init__()
        if threshold < 0:
            raise ValueError("The threshold must be greater than or equal to 0")
        self.threshold = threshold
        self.variance = None
 
    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Estimates the variance of each feature.
 
        Parameters
        ----------
        dataset: Dataset
            The Dataset object to fit the transformer to
 
        Returns
        -------
        self: VarianceThreshold
            The fitted transformer
        """
        self.variance = np.var(dataset.X, axis=0)
        return self
 
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the features whose variance is greater than the threshold.
 
        Parameters
        ----------
        dataset: Dataset
            The Dataset object to transform
 
        Returns
        -------
        dataset: Dataset
            A new Dataset object with the selected features only
        """
        mask = self.variance > self.threshold
        X_new = dataset.X[:, mask]
        features_new = [feature for feature, keep in zip(dataset.features, mask) if keep]
 
        return Dataset(X=X_new, y=dataset.y, features=features_new, label=dataset.label)
    