import pickle
import os
import json
import math
from path import Path


CURRENT_DIR = Path(__file__).parent.abspath()
ARGS_DICT = json.load(open(CURRENT_DIR / "args.json", "r"))
CLIENT_NUM_IN_EACH_PICKLES = ARGS_DICT["client_num_in_each_pickles"]
DATASET_DIR = (
    CURRENT_DIR
    if ARGS_DICT["dataset_dir"] is None
    else Path(ARGS_DICT["dataset_dir"]).abspath()
)


def get_dataset(dataset: str, client_id):
    PICKLES_DIR = DATASET_DIR / dataset / "pickles"
    if os.path.isdir(PICKLES_DIR) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    PICKLE_PATH = (
        PICKLES_DIR / f"{math.floor(client_id / CLIENT_NUM_IN_EACH_PICKLES)}.pkl"
    )
    with open(PICKLE_PATH, "rb") as f:
        subset = pickle.load(f)
    client_dataset = subset[client_id % CLIENT_NUM_IN_EACH_PICKLES]
    return client_dataset


def get_client_id_indices(dataset):
    dataset_pickles_path = DATASET_DIR / dataset / "pickles"
    with open(dataset_pickles_path / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    return (seperation["train"], seperation["test"], seperation["total"])
