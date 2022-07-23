import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import Dataset


def dirichlet_distribution(
    ori_dataset: Dataset,
    target_dataset: Dataset,
    num_clients: int,
    alpha: float,
    transform=None,
    target_transform=None,
) -> Tuple[List[Dataset], Dict]:
    NUM_CLASS = len(ori_dataset.classes)
    MIN_SIZE = 0
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    if not isinstance(ori_dataset.targets, np.ndarray):
        ori_dataset.targets = np.array(ori_dataset.targets, dtype=np.int64)
    idx = [np.where(ori_dataset.targets == i)[0] for i in range(NUM_CLASS)]

    while MIN_SIZE < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASS):
            np.random.shuffle(idx[k])
            distributions = np.random.dirichlet(np.repeat(alpha, num_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(ori_dataset) / num_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
            ]
            MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            np.random.shuffle(idx_batch[i])
            X[i] = ori_dataset.data[idx_batch[i]]
            Y[i] = ori_dataset.targets[idx_batch[i]]

    datasets = [
        target_dataset(
            data=X[j],
            targets=Y[j],
            transform=transform,
            target_transform=target_transform,
        )
        for j in range(num_clients)
    ]
    return datasets
