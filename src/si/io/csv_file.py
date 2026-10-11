import pandas as pd
from si.data.dataset import Dataset
 
 
def read_csv(filename: str, sep: str = ',', features: bool = True, label: bool = True) -> Dataset:
    """
    Reads a csv file and returns a Dataset object.
    The first line of the file is always read as the header (the column names).
 
    Parameters
    ----------
    filename: str
        Path to the csv file
    sep: str
        Value separator
    features: bool
        Whether the feature names (taken from the header) are kept in the Dataset.
        If False, the Dataset generates the names "feat_0", "feat_1", ...
    label: bool
        Whether the file has a label (y). If so, it is assumed to be the last column
 
    Returns
    -------
    Dataset
    """
    data = pd.read_csv(filename, sep=sep)
 
    if label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        feature_names = list(data.columns[:-1]) if features else None
        label_name = str(data.columns[-1])
    else:
        X = data.to_numpy()
        y = None
        feature_names = list(data.columns) if features else None
        label_name = None
 
    return Dataset(X=X, y=y, features=feature_names, label=label_name)
 
 
def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = True, label: bool = True) -> None:
    """
    Writes a Dataset object to a csv file.
    The first line of the file is always the header (the column names).
 
    Parameters
    ----------
    filename: str
        Path to the csv file
    dataset: Dataset
        The Dataset object to write
    sep: str
        Value separator
    features: bool
        Whether to use the feature names as column names.
        If False, the feature columns are named 0, 1, 2, ...
    label: bool
        Whether to write the label (y) as the last column
    """
    df = pd.DataFrame(dataset.X, columns=dataset.features if features else None)
 
    if label and dataset.has_label():
        df[dataset.label] = dataset.y
 
    df.to_csv(filename, sep=sep, index=False)
