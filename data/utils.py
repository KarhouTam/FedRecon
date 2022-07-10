import pickle
import os
from dataset import MNISTDataset, CIFARDataset
from path import Path

DATASET_DICT = {
    "mnist": MNISTDataset,
    "cifar10": CIFARDataset,
}
CURRENT_DIR = Path(__file__).parent.abspath()


def get_dataset(dataset: str, client_id):
    pickles_dir = CURRENT_DIR / dataset / "pickles"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    with open(pickles_dir / str(client_id) + ".pkl", "rb") as f:
        client_dataset: DATASET_DICT[dataset] = pickle.load(f)
    return client_dataset


def get_client_id_indices(dataset):
    dataset_pickles_path = CURRENT_DIR / dataset / "pickles"
    with open(dataset_pickles_path / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    return (seperation["train"], seperation["test"], seperation["total"])

