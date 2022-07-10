from collections import OrderedDict
import sys

sys.path.append("../")

import torch
from data.utils import get_dataset
from copy import deepcopy
from utils import train_with_logging
from torch.utils.data import DataLoader, random_split

SEEN_CLIENTS_ID = set()


class FedAvgTrainer:
    def __init__(self, args, model, logger):
        """
        The function takes in the arguments, the model, and the logger, and then it initializes the
        device, the local epochs, the logger, the model, the batch size, the validation set ratio, the
        dataset, the criterion, and the optimizer
        
        Args:
          args: the arguments passed to the main program
          model: The model that we want to train.
          logger: a logger object to log the training process
        """
        if args.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.local_epochs = args.pers_epochs
        self.logger = logger
        self.model = deepcopy(model)
        self.batch_size = args.batch_size
        self.valset_ratio = args.valset_ratio
        self.dataset = args.dataset
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimzier = torch.optim.SGD(self.model.parameters(), lr=args.pers_lr)

        self.id = None
        self.weight = 0
        self.train_set_dataloader = None
        self.val_set_dataloader = None
    def train(self, client_id, model_params, have_seen=None, validation=False):
        """
        The function takes in the model parameters, splits the dataset, trains the model, and returns
        the difference between the model parameters before and after training
        
        Args:
          client_id: the id of the client
          model_params: the model parameters that the client will use to train on
          have_seen: the number of samples the client has seen so far
          validation: whether to use validation set or not. Defaults to False
        
        Returns:
          The difference between the model parameters before and after training, and the weight of the
        client.
        """
        self.id = client_id

        self.model.load_state_dict(model_params, strict=False)

        self.split_dataset()

        model_diff = deepcopy(OrderedDict(self.model.named_parameters()))

        train_with_logging(self, validation)()

        with torch.no_grad():
            for frz_p, updated_p in zip(model_diff.values(), self.model.parameters()):
                frz_p.sub_(updated_p)

        return model_diff, self.weight

    def _train(self):
        """
        > For each epoch, for each batch, get the logit and loss, zero the gradients, backpropagate the
        loss, and update the weights
        """
        for _ in range(self.local_epochs):
            for x, y in self.train_set_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimzier.zero_grad()
                loss.backward()
                self.optimzier.step()

    def eval(self, model_params, client_id):
        """
        The function takes in a model, a client id, and a dataset, and returns the validation loss of
        the model on the client's validation set
        
        Args:
          model_params: the model parameters to be evaluated
          client_id: the id of the client
        
        Returns:
          The model parameters are being returned.
        """
        self.model.load_state_dict(model_params, strict=False)
        self.id = client_id
        self.split_dataset()
        return train_with_logging(self, validation=True)()

    def split_dataset(self):
        """
        The function takes in a dataset, splits it into a training set and a validation set
        """
        dataset = get_dataset(self.dataset, self.id)
        num_val_samples = int(self.valset_ratio * len(dataset))
        num_train_samples = len(dataset) - num_val_samples
        train_set, val_set = random_split(dataset, [num_train_samples, num_val_samples])
        self.train_set_dataloader = DataLoader(train_set, self.batch_size)
        self.val_set_dataloader = DataLoader(val_set, self.batch_size)
        self.weight = num_train_samples
