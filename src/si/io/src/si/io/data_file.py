import numpy as np
from si.data.dataset import Dataset

def read_data_file(filename: str, sep: str = ' ', label: bool = True) -> Dataset:
    data = np.genfromtxt(filename, delimiter=sep)
    
    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None

    return Dataset(X=X, y=y)

def write_data_file(filename: str, dataset: Dataset, sep: str = ' ', label: bool = True) -> None:
    if label and dataset.has_label():
        data = np.hstack((dataset.X, dataset.y.reshape(-1, 1)))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)



    def dropna(self) -> 'Dataset':
        """Remove todas as amostras contendo pelo menos um valor NaN."""
        mask = ~np.isnan(self.X).any(axis=1)
        self.X = self.X[mask]
        if self.has_label():
            self.y = self.y[mask]
        return self

    def fillna(self, value) -> 'Dataset':
        """Substitui valores NaN por um valor fixo, média ou mediana por feature."""
        nan_mask = np.isnan(self.X)
        if not np.any(nan_mask):
            return self

        for j in range(self.X.shape[1]):
            col = self.X[:, j]
            col_nan = np.isnan(col)
            if np.any(col_nan):
                if value == "mean":
                    fill_val = np.nanmean(col)
                elif value == "median":
                    fill_val = np.nanmedian(col)
                else:
                    fill_val = float(value)
                col[col_nan] = fill_val
        return self

    def remove_by_index(self, index: int) -> 'Dataset':
        """Remove uma amostra pelo seu índice."""
        self.X = np.delete(self.X, index, axis=0)
        if self.has_label():
            self.y = np.delete(self.y, index, axis=0)
        return self