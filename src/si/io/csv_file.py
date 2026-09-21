import pandas as pd
import numpy as np
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ',', features: bool = True, label: bool = True) -> Dataset:
    data = pd.read_csv(filename, sep=sep)
    
    if features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        feature_names = list(data.columns[:-1])
        label_name = str(data.columns[-1])
    elif features and not label:
        X = data.to_numpy()
        y = None
        feature_names = list(data.columns)
        label_name = None
    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        feature_names = None
        label_name = None
    else:
        X = data.to_numpy()
        y = None
        feature_names = None
        label_name = None

    return Dataset(X=X, y=y, features=feature_names, label=label_name)

def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = True, label: bool = True) -> None:
    df_dict = {}
    
    if features and dataset.features is not None:
        for i, name in enumerate(dataset.features):
            df_dict[name] = dataset.X[:, i]
    else:
        for i in range(dataset.X.shape[1]):
            df_dict[f"feature_{i}"] = dataset.X[:, i]

    if label and dataset.has_label():
        label_title = dataset.label if dataset.label else "y"
        df_dict[label_title] = dataset.y

    df = pd.DataFrame(df_dict)
    df.to_csv(filename, sep=sep, index=False)