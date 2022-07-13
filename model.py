from torch import nn
from collections import OrderedDict

# NOTE:
# If you wanna build your own model, please inherit your model class from MetaModel,
# and using split_model([local layers defined by you]) at the end of __init__()
class MetaModel(nn.Module):
    def __init__(self,):
        super(MetaModel, self).__init__()
        self.layers_name = []
        self.local_layers = []
        self.global_layers = []

    def _split_model(self, local_layers=[]):      
        self.layers_name = list(
            set([name.split(".")[0] for name, _ in self.named_parameters()])
        )
        self.local_layers = local_layers
        self.global_layers = list(set(self.layers_name) - set(self.local_layers))

    def global_params(self, requires_name=False, data_only=False):
        return self._specific_parameters(self.global_layers, requires_name, data_only)

    def local_params(self, requires_name=False, data_only=False):
        return self._specific_parameters(self.local_layers, requires_name, data_only)

    def _specific_parameters(self, layer_list, requires_name=False, data_only=False):
        """
        It returns a list of parameters of the layers in the layer_list.
        
        Args:
          layer_list: a list of strings, each string is the name of a layer in the model
          requires_name: If True, returns a dictionary of parameters with their names as keys. If False,
        returns a list of parameters. Defaults to False
          data_only: If True, returns the data of the parameters, otherwise returns the parameters
        themselves. Defaults to False
        
        Returns:
          The parameters of the layers in the layer_list.
        """
        if requires_name:
            param_dict = OrderedDict()
            for name, param in self.named_parameters():
                if name.split(".")[0] in layer_list:
                    if data_only:
                        param_dict[name] = param.detach().clone().data
                    else:
                        param_dict[name] = param
            return param_dict
        else:
            param_list = []
            for name, param in self.named_parameters():
                if name.split(".")[0] in layer_list:
                    if data_only:
                        param_list.append(param.detach().clone().data)
                    else:
                        param_list.append(param)
            return param_list


class CNN_MNIST(MetaModel):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.ReLU(True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        # NOTE: layer's name must be identical to the corresponding layer variance's name
        self._split_model(local_layers=["fc1", "fc2"])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        return x


class CNN_CIFAR10(MetaModel):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.ReLU(True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self._split_model(local_layers=["fc1", "fc2", "fc3"])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        return x


MODEL_DICT = {"mnist": CNN_MNIST, "cifar10": CNN_CIFAR10}


def get_model(dataset):
    return MODEL_DICT[dataset]()
