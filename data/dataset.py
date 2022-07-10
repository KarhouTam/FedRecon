import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, subset=None, data=None, targets=None) -> None:
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
            self.targets = torch.stack(
                list(map(lambda tup: torch.tensor(tup[1]), subset))
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


class CIFARDataset(Dataset):
    def __init__(self, subset=None, data=None, targets=None) -> None:
        if (data is not None) and (targets is not None):
            self.data = data.reshape(-1, 3, 32, 32)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
            self.targets = torch.stack(
                list(map(lambda tup: torch.tensor(tup[1]), subset))
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)
