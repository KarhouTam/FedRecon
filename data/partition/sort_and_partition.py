import torch
from fedlab.utils.dataset.slicing import noniid_slicing
from torch.utils.data import Dataset


def randomly_alloc_classes(
    raw_dataset: Dataset, target_dataset: Dataset, num_clients, num_classes
):
    """
    It takes a dataset and a number of clients, and returns a list of datasets, each of which is a
    subset of the original dataset, and each of which has a different set of classes
    
    Args:
      raw_dataset (Dataset): the original dataset
      target_dataset (Dataset): the dataset that we want to split into clients.
      num_clients: number of clients
      num_classes: number of classes in the dataset
    
    Returns:
      A list of datasets, each of which has a subset of the classes.
    """
    idxs = noniid_slicing(raw_dataset, num_clients, num_clients * num_classes)
    datasets = []
    for indices in idxs.values():
        datasets.append(target_dataset([raw_dataset[i] for i in indices]))
    for i, ds in enumerate(datasets):
        print(f"client [{i}] has classes: {torch.unique(ds.targets).numpy()}")
    return datasets
