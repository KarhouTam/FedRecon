import torch
from fedlab.utils.dataset.slicing import noniid_slicing
from torch.utils.data import Dataset


def randomly_alloc_classes(
    ori_dataset: Dataset,
    target_dataset: Dataset,
    num_clients,
    num_classes,
    transform=None,
    target_transform=None,
):
    """
  > Given a dataset, randomly allocate classes to clients
  
  Args:
    ori_dataset (Dataset): the original dataset
    target_dataset (Dataset): the dataset you want to use.
    num_clients: the number of clients
    num_classes: number of classes in the dataset
    transform: a function that takes in an image and returns a transformed image
    target_transform: a function/transform that takes in the target and transforms it.
  
  Returns:
    A list of datasets, each dataset is a subset of the original dataset.
  """
    idxs = noniid_slicing(ori_dataset, num_clients, num_clients * num_classes)
    datasets = []
    for indices in idxs.values():
        datasets.append(
            target_dataset(
                [ori_dataset[i] for i in indices],
                transform=transform,
                target_transform=target_transform,
            )
        )
    for i, ds in enumerate(datasets):
        print(f"client [{i}]] has classes: {torch.unique(ds.targets).numpy()}")
    return datasets
