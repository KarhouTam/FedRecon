import sys

sys.path.append("data")

import torch
import os
import random
from rich.console import Console
from algorithms import FedReconTrainer, FedAvgTrainer
from utils import LOG_DIR, fix_random_seed, get_args
from data.utils import get_client_id_indices
from collections import OrderedDict
from model import get_model
from copy import deepcopy

SEEN_CLIENTS_ID = set()
TRAINER_DICT = {"fedrecon": FedReconTrainer, "fedavg": FedAvgTrainer}

if __name__ == "__main__":

    args = get_args()
    fix_random_seed(args.seed)

    if os.path.isdir("clients"):
        os.system("rm -rf clients")
    os.mkdir("clients")

    if not os.path.isdir("log"):
        os.mkdir("log")

    logger = Console(record=True)

    log_path = (
        LOG_DIR / args.algo
        + "_"
        + args.dataset
        + "_"
        + str(args.global_epochs)
        + "_"
        + str(args.client_num_per_round)
        + ".html"
    )

    if args.gpu != 0 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_clients, test_clients, num_clients = get_client_id_indices(args.dataset)
    model = get_model(args.dataset).to(device)
    trainer = TRAINER_DICT[args.algo](args, model, logger)

    logger.log(f"Arguments:\n{dict(args._get_kwargs())}", justify="left")
    # training
    logger.log("=" * 30, "TRAINING", "=" * 30, style="bold yellow")
    for _ in range(args.global_epochs):
        diff_list = []
        weight_list = []
        selected_clients = random.sample(train_clients, args.client_num_per_round)
        for client_id in selected_clients:
            if args.algo == "fedrecon":
                model_params = model.global_params(requires_name=True, data_only=True)
            else:
                model_params = deepcopy(OrderedDict(model.named_parameters()))
            model_diff, weight = trainer.train(
                client_id=client_id,
                model_params=model_params,
                have_seen=client_id in SEEN_CLIENTS_ID,
                validation=args.eval_while_training,
            )
            diff_list.append(model_diff)
            weight_list.append(weight)

            SEEN_CLIENTS_ID.add(client_id)
        logger.log("=" * 70, style="bold yellow")

        # aggregation
        with torch.no_grad():
            # calculate weights
            weight_sum = sum(weight_list)
            weight_list = list(map(lambda w: w / weight_sum, weight_list))
            for diff, weight in zip(diff_list, weight_list):
                for param in diff.values():
                    param.data = weight * param.data

            # aggregate model params
            weighted_diff = OrderedDict()
            for diff in diff_list:
                for layer_name, param in diff.items():
                    if layer_name not in weighted_diff:
                        weighted_diff[layer_name] = param
                    else:
                        weighted_diff[layer_name] += param

            # update global model
            model_params = OrderedDict(model.named_parameters())
            for layer_name, diff in weighted_diff.items():
                model_params[layer_name].sub_(args.server_lr * diff)

    # evaluation
    logger.log("=" * 30, "EVALUATION", "=" * 30, style="bold blue")
    if args.algo == "fedrecon":
        model_params = model.global_params(requires_name=True, data_only=True)
    else:
        model_params = deepcopy(OrderedDict(model.named_parameters()))
    global_params = model.global_params(requires_name=True, data_only=True)
    all_results = {"loss_before": 0, "loss_after": 0, "acc_before": 0, "acc_after": 0}
    for client_id in test_clients:
        result = trainer.eval(client_id=client_id, model_params=model_params)
        all_results["loss_before"] += result["loss_before"]
        all_results["loss_after"] += result["loss_after"]
        all_results["acc_before"] += result["acc_before"]
        all_results["acc_after"] += result["acc_after"]

    all_results["loss_before"] /= len(test_clients)
    all_results["loss_after"] /= len(test_clients)
    all_results["acc_before"] /= len(test_clients)
    all_results["acc_after"] /= len(test_clients)

    logger.log("=" * 30, "RESULTS", "=" * 30, style="bold green")

    logger.log(
        "loss: {:.4f} -> {:.4f}".format(
            all_results["loss_before"], all_results["loss_after"]
        ),
        style="bold red",
    )
    logger.log(
        "acc: {:.2f}% -> {:.2f}%".format(
            all_results["acc_before"], all_results["acc_after"]
        ),
        style="bold blue",
    )

    logger.save_html(log_path)
