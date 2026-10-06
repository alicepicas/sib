import pandas as pd
from si.data.dataset import Dataset

 
def read_csv(filename: str, sep: str = ',', features: bool = True, label: bool = True) -> Dataset:
    """
    Reads a csv file and returns a Dataset object.
 
    Parameters
    ----------
    filename: str
        Path to the csv file
    sep: str
        Value separator
    features: bool
        Whether the file has a header with the feature names
    label: bool
        Whether the file has a label (y). If so, it is assumed to be the last column
 
    Returns
    -------
    Dataset
    """

    data = pd.read_csv(filename, sep=sep, header=0 if features else None)
 
    if label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        feature_names = list(data.columns[:-1]) if features else None
        label_name = str(data.columns[-1]) if features else None
    else:
        X = data.to_numpy()
        y = None
        feature_names = list(data.columns) if features else None
        label_name = None
 
    return Dataset(X=X, y=y, features=feature_names, label=label_name)
 
 
def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = True, label: bool = True) -> None:
    """
    Writes a Dataset object to a csv file.
 
    Parameters
    ----------
    filename: str
        Path to the csv file
    dataset: Dataset
        The Dataset object to write
    sep: str
        Value separator
    features: bool
        Whether to write a header with the feature names
    label: bool
        Whether to write the label (y) as the last column
    """
    
    df = dataset.to_dataframe()
 
    if not label and dataset.has_label():
        df = df.drop(columns=dataset.label)
 
    df.to_csv(filename, sep=sep, index=False, header=features)
    
