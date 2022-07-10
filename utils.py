import torch
import random
import numpy as np
from argparse import ArgumentParser, Namespace
from path import Path


LOG_DIR = Path(__file__).parent.abspath() / "log"


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--algo", type=str, choices=["fedavg", "fedrecon"], default="fedrecon"
    )
    parser.add_argument("--global_epochs", type=int, default=20)
    parser.add_argument("--pers_epochs", type=int, default=1)
    parser.add_argument("--recon_epochs", type=int, default=1)
    parser.add_argument("--pers_lr", type=float, default=1e-2)
    parser.add_argument("--recon_lr", type=float, default=1e-2)
    parser.add_argument("--server_lr", type=float, default=1.0)
    parser.add_argument("--client_num_per_round", type=int, default=5)
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
    )
    parser.add_argument("--no_split", type=int, default=0)
    parser.add_argument("--eval_while_training", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--valset_ratio", type=float, default=0.1)
    return parser.parse_args()


def fix_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, dataloader, criterion, device=torch.device("cpu")):
    model.eval()
    total_loss = 0
    num_samples = 0
    acc = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logit = model(x)
        total_loss += criterion(logit, y)
        pred = torch.softmax(logit, -1).argmax(-1)
        acc += torch.eq(pred, y).int().sum()
        num_samples += y.size(-1)
    model.train()
    return total_loss, acc / num_samples


def train_with_logging(trainer, validation=False):
    def training_func(*args, **kwargs):
        if validation:
            loss_before, acc_before = evaluate(
                trainer.model,
                trainer.val_set_dataloader,
                trainer.criterion,
                trainer.device,
            )
        trainer._train(*args, **kwargs)
        if validation:
            loss_after, acc_after = evaluate(
                trainer.model,
                trainer.val_set_dataloader,
                trainer.criterion,
                trainer.device,
            )
            trainer.logger.log(
                "client [{}]   [red]loss:{:.4f} -> {:.4f}    [blue]acc:{:.2f}% -> {:.2f}%".format(
                    trainer.id,
                    loss_before,
                    loss_after,
                    (acc_before.item() * 100.0),
                    (acc_after.item() * 100.0),
                )
            )

        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "acc_before": (acc_before.item() * 100.0),
            "acc_after": (acc_after.item() * 100.0),
        }

    return training_func

