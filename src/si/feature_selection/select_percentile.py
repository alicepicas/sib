from typing import Callable
import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
 
 
class SelectPercentile(Transformer):
    """
    Selects the features with the highest scores, according to a percentile.
    The score of each feature is given by a scoring function (the F value of f_classification by default).
 
    Parameters
    ----------
    score_func: Callable
        The scoring function. It receives a Dataset and returns the F and p values of each feature
    percentile: int
        The percentage of features to select (e.g., 40 for 40%)
 
    Attributes
    ----------
    F: np.ndarray
        The F value of each feature, estimated by the scoring function
    p: np.ndarray
        The p value of each feature, estimated by the scoring function
    """
 
    def __init__(self, score_func: Callable = f_classification, percentile: int = 40):
        """
        Initialize the SelectPercentile transformer.
 
        Parameters
        ----------
        score_func: Callable
            The scoring function. It receives a Dataset and returns the F and p values of each feature
        percentile: int
            The percentage of features to select (e.g., 40 for 40%)
        """
        super().__init__()
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
 
    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimates the F and p values of each feature using the scoring function.
 
        Parameters
        ----------
        dataset: Dataset
            The Dataset object to fit the transformer to
 
        Returns
        -------
        self: SelectPercentile
            The fitted transformer
        """
        self.F, self.p = self.score_func(dataset)
        return self
 
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the features with the highest F values, according to the percentile.
        The threshold is the F value at the (100 - percentile) percentile. The features with an F value
        above the threshold are selected and, if the percentage is not yet reached, the first features
        whose F value equals the threshold (ties) are also selected.
 
        Parameters
        ----------
        dataset: Dataset
            The Dataset object to transform
 
        Returns
        -------
        dataset: Dataset
            A new Dataset object with the selected features only
        """
        n_features = dataset.X.shape[1]
        n_select = int(n_features * self.percentile / 100)
 
        threshold = np.percentile(self.F, 100 - self.percentile)
        mask = self.F > threshold
 
        ties = np.where(self.F == threshold)[0]
        n_missing = n_select - mask.sum()
        if n_missing > 0:
            mask[ties[:n_missing]] = True
 
        X_new = dataset.X[:, mask]
        features_new = [feature for feature, keep in zip(dataset.features, mask) if keep]
 
        return Dataset(X=X_new, y=dataset.y, features=features_new, label=dataset.label)
    