from typing import Callable
import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
 
 
class SelectKBest(Transformer):
    """
    Selects the k features with the highest scores.
    The score of each feature is given by a scoring function (the F value of f_classification by default).
 
    Parameters
    ----------
    score_func: Callable
        The scoring function. It receives a Dataset and returns the F and p values of each feature
    k: int
        The number of features to select
 
    Attributes
    ----------
    F: np.ndarray
        The F value of each feature, estimated by the scoring function
    p: np.ndarray
        The p value of each feature, estimated by the scoring function
    """
 
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        Initialize the SelectKBest transformer.
 
        Parameters
        ----------
        score_func: Callable
            The scoring function. It receives a Dataset and returns the F and p values of each feature
        k: int
            The number of features to select
        """
        super().__init__()
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None
 
    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Estimates the F and p values of each feature using the scoring function.
 
        Parameters
        ----------
        dataset: Dataset
            The Dataset object to fit the transformer to
 
        Returns
        -------
        self: SelectKBest
            The fitted transformer
        """
        self.F, self.p = self.score_func(dataset)
        return self
 
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the k features with the highest F values, keeping their original order.
 
        Parameters
        ----------
        dataset: Dataset
            The Dataset object to transform
 
        Returns
        -------
        dataset: Dataset
            A new Dataset object with the selected features only
        """
        best_idx = np.sort(np.argsort(-self.F, kind='stable')[:self.k])
        X_new = dataset.X[:, best_idx]
        features_new = [dataset.features[i] for i in best_idx]
 
        return Dataset(X=X_new, y=dataset.y, features=features_new, label=dataset.label)