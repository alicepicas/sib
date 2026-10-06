import numpy as np
from si.data.dataset import Dataset
 
 
def read_data_file(filename: str, sep: str = ' ', label: bool = True) -> Dataset:
    """
    Reads a data file (without header) and returns a Dataset object.
 
    Parameters
    ----------
    filename: str
        Path to the data file
    sep: str
        Value separator used in the file
    label: bool
        Whether the file has a label (y). If so, it is assumed to be the last column
 
    Returns
    -------
    Dataset
    """

    data = np.genfromtxt(filename, delimiter=sep)
 
    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None
 
    return Dataset(X=X, y=y)
 
 
def write_data_file(filename: str, dataset: Dataset, sep: str = ' ', label: bool = True) -> None:
    """
    Writes a Dataset object to a data file (without header).
 
    Parameters
    ----------
    filename: str
        Path to the data file
    dataset: Dataset
        The Dataset object to write
    sep: str
        Value separator
    label: bool
        Whether to write the label (y) as the last column
    """
    
    if label and dataset.has_label():
        data = np.hstack((dataset.X, dataset.y.reshape(-1, 1)))
    else:
        data = dataset.X
 
    np.savetxt(filename, data, delimiter=sep)
    
