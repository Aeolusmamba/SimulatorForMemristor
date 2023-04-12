import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist(batch_size, data_root="public_dataset", train=True, test=True, num_workers=1):
    data_root = os.path.join(os.path.dirname(__file__), data_root)
    data_set = []
    if train:
        train_loader = DataLoader(datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(28),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(.5, .5)
            ])
        ), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        data_set.append(train_loader)
    if test:
        test_loader = DataLoader(
            datasets.MNIST(
                root=data_root,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(28),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(.5, .5)
                ])
            ), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        data_set.append(test_loader)
        data_set = data_set[0] if len(data_set) == 1 else data_set
        return data_set
