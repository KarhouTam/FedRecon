import sys

sys.path.append("/")
sys.path.append("data")

import torch
import pickle
from data.utils import get_dataset
from copy import deepcopy
from utils import train_with_logging
from torch.utils.data import random_split, DataLoader
from path import Path

PROJECT_DIR = Path(__file__).parent.parent.abspath()
CLIENTS_DIR = PROJECT_DIR / "clients"


class FedReconTrainer:
    def __init__(self, args, model, logger):
        """
        The function initializes the parameters of the model, and also initializes the dataloaders for
        the support set, query set, and validation set
        
        Args:
          args: the arguments passed to the main script
          model: the model to be trained
          logger: a logger object to log the training process
        """
        if args.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.recon_epochs = args.recon_epochs
        self.pers_epochs = args.pers_epochs
        self.logger = logger
        self.model = deepcopy(model)
        self.backup_local_params = self.model.local_params(
            requires_name=True, data_only=True
        )
        self.batch_size = args.batch_size
        self.valset_ratio = args.valset_ratio
        self.dataset = args.dataset
        self.criterion = torch.nn.CrossEntropyLoss()
        self.recon_lr = args.recon_lr
        self.pers_lr = args.pers_lr
        self.no_split = args.no_split

        self.id = None
        self.support_set_dataloader = None
        self.query_set_dataloader = None
        self.val_set_dataloader = None
        self.weight = 0

    def train(self, client_id, model_params, have_seen=False, validation=False):
        """
        The function takes in the client id, the model parameters, and a boolean value indicating whether
        the client has been trained before. If the client has been trained before, the function loads
        the client data. Otherwise, the function splits the dataset and loads the backup local
        parameters. The function then calculates the model difference and trains the model with logging.
        The function then calculates the pseudo gradients and saves the client data. The function returns
        the model difference and the weight
        
        Args:
          client_id: the id of the client
          model_params: the global model parameters
          have_seen: whether the client has been trained before. Defaults to False
          validation: whether to use the validation set or not. Defaults to False
        
        Returns:
          The model_diff is the difference between the global model and the local model.
        """
        self.id = client_id
        self.model.load_state_dict(model_params, strict=False)

        if have_seen:
            self.load_client_data(client_id)
        else:
            self.split_dataset()
            self.model.load_state_dict(self.backup_local_params, strict=False)

        self.id = client_id
        model_diff = self.model.global_params(requires_name=True, data_only=True)

        train_with_logging(self, validation)()

        # calculate the pseudo gradients
        with torch.no_grad():
            for frz_p, updated_p in zip(
                model_diff.values(), self.model.global_params()
            ):
                frz_p.sub_(updated_p)

        self.save_client_data()

        return model_diff, self.weight

    def _train(self):
        """
        > For each epoch, we first train the local model on the support set, then we train the global model
        on the query set
        """
        # reconstruction phase
        for _ in range(self.recon_epochs):
            for x, y in self.support_set_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                gradients = torch.autograd.grad(loss, self.model.local_params())
                for param, grad in zip(self.model.local_params(), gradients):
                    param.data -= self.recon_lr * grad
        # personalzation phase
        for _ in range(self.pers_epochs):
            for x, y in self.query_set_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                gradients = torch.autograd.grad(loss, self.model.global_params())
                for param, grad in zip(self.model.global_params(), gradients):
                    param.data -= self.pers_lr * grad

    def eval(self, model_params, client_id):
        """
        The function takes in the model parameters and the client id, and then loads the model
        parameters into the model, and then loads the backup local parameters into the model, and then
        splits the dataset, and then returns the result of the train_with_logging function
        
        Args:
          model_params: the model parameters to be evaluated
          client_id: the id of the client
        
        Returns:
          The return value is the validation loss.
        """
        self.id = client_id

        self.model.load_state_dict(model_params, strict=False)
        self.model.load_state_dict(self.backup_local_params, strict=False)

        self.split_dataset()

        return train_with_logging(self, validation=True)()

    def split_dataset(self):
        """
        The function splits the dataset into training, validation and test sets.
        """

        dataset = get_dataset(self.dataset, self.id)

        num_val_samples = int(self.valset_ratio * len(dataset))
        num_train_samples = len(dataset) - num_val_samples

        training_set, val_set = random_split(
            dataset, [num_train_samples, num_val_samples]
        )
        if self.no_split:
            num_support_samples = num_query_samples = num_train_samples
            support_set = query_set = training_set
        else:
            # query set's size is set same as the support set's by default.
            num_support_samples = int(num_train_samples / 2)
            num_query_samples = num_train_samples - num_support_samples
            support_set, query_set = random_split(
                training_set, [num_support_samples, num_query_samples]
            )

        self.support_set_dataloader = DataLoader(support_set, self.batch_size)
        self.query_set_dataloader = DataLoader(query_set, self.batch_size)
        self.val_set_dataloader = DataLoader(val_set, self.batch_size)
        self.weight = num_query_samples

    def save_client_data(self):
        """
        It saves the client's data, weight, and local parameters to a pickle file
        """
        local_params = self.model.local_params(requires_name=True, data_only=True)
        pkl_path = CLIENTS_DIR / f"{self.id}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "support": self.support_set_dataloader,
                    "query": self.query_set_dataloader,
                    "val": self.val_set_dataloader,
                    "weight": self.weight,
                    "local_params": local_params,
                },
                f,
            )

    def load_client_data(self, client_id):
        """
        It loads the client data from a pickle file, and then sets the support, query, and validation
        dataloaders, as well as the weight and local parameters of the model
        
        Args:
          client_id: the id of the client we want to load data for
        """
        pkl_path = CLIENTS_DIR / f"{client_id}.pkl"
        with open(pkl_path, "rb") as f:
            client_data = pickle.load(f)
        self.support_set_dataloader = client_data["support"]
        self.query_set_dataloader = client_data["query"]
        self.val_set_dataloader = client_data["val"]
        self.weight = client_data["weight"]
        local_params = client_data["local_params"]
        self.model.load_state_dict(local_params, strict=False)

