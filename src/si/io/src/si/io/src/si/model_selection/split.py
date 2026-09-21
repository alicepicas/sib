import numpy as np
from typing import Tuple
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.X.shape[0]
    permutations = np.random.permutation(n_samples)
    
    n_test = int(n_samples * test_size)
    test_idx = permutations[:n_test]
    train_idx = permutations[n_test:]

    train_dataset = Dataset(
        X=dataset.X[train_idx],
        y=dataset.y[train_idx] if dataset.has_label() else None,
        features=dataset.features,
        label=dataset.label
    )
    test_dataset = Dataset(
        X=dataset.X[test_idx],
        y=dataset.y[test_idx] if dataset.has_label() else None,
        features=dataset.features,
        label=dataset.label
    )

    return train_dataset, test_dataset

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    if random_state is not None:
        np.random.seed(random_state)

    unique_labels, counts = np.unique(dataset.y, return_counts=True)
    train_indices = []
    test_indices = []

    for label in unique_labels:
        label_idx = np.where(dataset.y == label)[0]
        np.random.shuffle(label_idx)
        
        n_test = int(len(label_idx) * test_size)
        test_indices.extend(label_idx[:n_test])
        train_indices.extend(label_idx[n_test:])

    train_dataset = Dataset(
        X=dataset.X[train_indices],
        y=dataset.y[train_indices],
        features=dataset.features,
        label=dataset.label
    )
    test_dataset = Dataset(
        X=dataset.X[test_indices],
        y=dataset.y[test_indices],
        features=dataset.features,
        label=dataset.label
    )

    return train_dataset, test_dataset