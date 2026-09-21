from scipy.stats import f_oneway
from si.data.dataset import Dataset

def f_classification(dataset: Dataset):
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    f_vals, p_vals = f_oneway(*groups)
    return f_vals, p_vals