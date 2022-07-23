import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            self.data = data.float().unsqueeze(1)

            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            self.targets = targets.long()
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else ToTensor()(tup[0]),
                        subset,
                    )
                )
            ).float()
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            ).long()
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], int(self.targets[index])

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)


class CIFARDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            self.data = data.float().reshape(-1, 3, 32, 32)

            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            self.targets = targets.long()
        elif subset is not None:
            self.data = (
                torch.stack(
                    list(
                        map(
                            lambda tup: tup[0]
                            if isinstance(tup[0], torch.Tensor)
                            else ToTensor()(tup[0]),
                            subset,
                        )
                    )
                )
                .float()
                .reshape(-1, 3, 32, 32)
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            ).long()

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return img, targets

    def __len__(self):
        return len(self.targets)

