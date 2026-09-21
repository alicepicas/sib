import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile(Transformer):
    def __init__(self, score_func=f_classification, percentile: int = 40):
        super().__init__()
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        n_features = dataset.X.shape[1]
        k = int(np.ceil(n_features * (self.percentile / 100.0)))
        
        # Obter os k índices com maiores valores de F
        top_k_idx = np.argsort(self.F)[-k:]
        
        X_trans = dataset.X[:, top_k_idx]
        features_trans = [dataset.features[i] for i in top_k_idx] if dataset.features else None
        
return Dataset(X=X_trans, y=dataset.y, features=features_trans, label=dataset.label)