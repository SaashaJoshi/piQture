# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Data Loader for MNIST images"""

import torch.utils.data
import torchvision
from torchvision import datasets


def load_mnist_data():
    """
    Loads MNIST dataset from PyTorch using DataLoader.
    Return:
        Train and Test DataLoader objects.
    """

    mnist_train = datasets.MNIST(
        root="data_loader/mnist_data",
        train=True,
        download=False,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(2)]
        ),
    )

    mnist_test = datasets.MNIST(
        root="data_loader/mnist_data",
        train=False,
        download=False,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(2)]
        ),
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=mnist_train,
        batch_size=60000,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=mnist_test,
        batch_size=10000,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return mnist_train, mnist_test, train_dataloader, test_dataloader


def collate_fn(batch):
    """
    Batches the data according to the defined function.
    """
    new_batch = []
    for img, label in batch:
        item = img, label
        if label in (1, 7):
            new_batch.append(item)
    return torch.utils.data.default_collate(new_batch)
