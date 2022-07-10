from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset

# Codes below are modified from https://github.com/WonJoon-Yun/Non-IID-Dataset-Generator-Dirichlet
def dirichlet_distribution(
    ori_dataset: Dataset,
    target_dataset: Dataset,
    num_clients: int,
    alpha: int,
    transform=None,
    target_transform=None,
) -> List[Dataset]:
    """
    `dirichlet_distribution` takes a dataset, a number of clients, a number of classes, and a
    hyperparameter alpha, and returns a list of datasets, each of which is a subset of the original
    dataset
    
    Args:
      ori_dataset (Dataset): the original dataset
      target_dataset (Dataset): the dataset that you want to split
      num_clients (int): the number of clients
      alpha (int): the parameter of the Dirichlet distribution.
      transform: A function/transform that takes in an PIL image or a numpy array and returns a transformed version. E.g,
    transforms.RandomCrop
      target_transform: A function/transform that takes in the target and transforms it.
    
    Returns:
      A list of datasets.
    """
    num_classes = len(ori_dataset.classes)
    if not isinstance(ori_dataset.targets, torch.Tensor):
        ori_dataset.targets = torch.tensor(ori_dataset.targets)
    if not isinstance(ori_dataset.data, torch.Tensor):
        ori_dataset.data = torch.tensor(ori_dataset.data)
    ori_dataset.data = ori_dataset.data.float()
    idx = [torch.where(ori_dataset.targets == i) for i in range(num_classes)]
    data = [ori_dataset.data[idx[i][0]] for i in range(num_classes)]
    label = [torch.ones(len(data[i]), dtype=torch.long) * i for i in range(num_classes)]
    s = np.random.dirichlet(np.ones(num_classes) * alpha, num_clients)
    data_dist = np.zeros((num_clients, num_classes))

    for j in range(num_clients):
        data_dist[j] = (
            (s[j] * len(data[0])).astype("int")
            / (s[j] * len(data[0])).astype("int").sum()
            * len(data[0])
        ).astype("int")
        data_num = data_dist[j].sum()
        data_dist[j][np.random.randint(low=0, high=num_classes)] += (
            len(data[0]) - data_num
        )
        data_dist = data_dist.astype("int")

    X = []
    Y = []
    for j in range(num_clients):
        x_data = []
        y_data = []
        for i in range(num_classes):
            if data_dist[j][i] != 0:
                d_index = np.random.randint(
                    low=0, high=len(data[i]), size=data_dist[j][i]
                )
                x_data.append(data[i][d_index])
                y_data.append(label[i][d_index])
        X.append(torch.cat(x_data))
        Y.append(torch.cat(y_data))
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
