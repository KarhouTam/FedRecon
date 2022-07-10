import sys

sys.path.append("../")
import os
import pickle
import numpy as np
import random
import torch
from path import Path
from argparse import ArgumentParser
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from dataset import MNISTDataset, CIFARDataset
from partition import dirichlet_distribution, randomly_alloc_classes

CURRENT_DIR = Path(__file__).parent.abspath()

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "cifar10": (CIFAR10, CIFARDataset),
}


MEAN = {
    "mnist": (0.1307,),
    "cifar10": (0.4914, 0.4822, 0.4465),
}

STD = {
    "mnist": (0.3015,),
    "cifar10": (0.2023, 0.1994, 0.2010),
}


def preprocess(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    num_train_clients = int(args.client_num_in_total * args.fraction)
    num_test_clients = args.client_num_in_total - num_train_clients
    dataset_dir = CURRENT_DIR / args.dataset
    pickles_dir = CURRENT_DIR / args.dataset / "pickles"
    transform = transforms.Compose(
        [transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),]
    )
    target_transform = None
    if not os.path.isdir(CURRENT_DIR / args.dataset):
        os.mkdir(CURRENT_DIR / args.dataset)
    if os.path.isdir(pickles_dir):
        os.system(f"rm -rf {pickles_dir}")
    os.mkdir(f"{pickles_dir}")

    ori_dataset, target_dataset = DATASET[args.dataset]
    trainset = ori_dataset(dataset_dir, train=True, download=True,)
    testset = ori_dataset(dataset_dir, train=False,)

    if args.alpha > 0:  # performing Dirichlet(alpha) partition
        all_trainsets = dirichlet_distribution(
            ori_dataset=trainset,
            target_dataset=target_dataset,
            num_clients=num_train_clients,
            alpha=args.alpha,
            transform=transform,
            target_transform=target_transform,
        )
        all_testsets = dirichlet_distribution(
            ori_dataset=testset,
            target_dataset=target_dataset,
            num_clients=num_test_clients,
            alpha=args.alpha,
            transform=transform,
            target_transform=target_transform,
        )
    else:
        classes = ori_dataset.classes if args.classes <= 0 else args.classes
        all_trainsets = randomly_alloc_classes(
            ori_dataset=trainset,
            target_dataset=target_dataset,
            num_clients=num_train_clients,
            num_classes=classes,
            transform=transform,
            target_transform=target_transform,
        )
        all_testsets = randomly_alloc_classes(
            ori_dataset=testset,
            target_dataset=target_dataset,
            num_clients=num_test_clients,
            num_classes=classes,
            transform=transform,
            target_transform=target_transform,
        )

    all_datasets = all_trainsets + all_testsets

    for client_id, dataset in enumerate(all_datasets):
        with open(pickles_dir / str(client_id) + ".pkl", "wb") as f:
            pickle.dump(dataset, f)
    with open(pickles_dir / "seperation.pkl", "wb") as f:
        pickle.dump(
            {
                "train": [i for i in range(num_train_clients)],
                "test": [i for i in range(num_train_clients, args.client_num_in_total)],
                "total": args.client_num_in_total,
            },
            f,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar10"], default="mnist",
    )
    parser.add_argument("--client_num_in_total", type=int, default=200)
    parser.add_argument(
        "--fraction", type=float, default=0.9, help="Propotion of train clients"
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=-1,
        help="Num of classes that one client's data belong to.",
    )
    parser.add_argument("--seed", type=int, default=0)

    ################# For dirichlet distribution only #################
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Only for control non-iid level while performing Dirichlet partition.",
    )
    ###################################################################
    args = parser.parse_args()
    preprocess(args)
