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

from __future__ import annotations
from typing import Union
import torch.utils.data
import torchvision
from torchvision import datasets


def load_mnist_dataset(
    labels: list = list(range(10)),
    batch_size: int = 64,
    size: Union[int, tuple] = 2,
    normalize_min: int = 0,
    normalize_max: int = 1,
):
    """
    Loads MNIST dataset from PyTorch using DataLoader.

    Args:
        labels (list): List of desired labels.
        batch_size (int, optional): Batch size for the dataset. Defaults to 64.
        size (int or tuple, optional): Size to which images will be resized. Defaults to 2.
            If an integer is provided, images will be resized to a
            square of that size.
            If a tuple of integers is provided, images will be resized to
            the specified height and width.
        normalize_min (float, optional): Minimum value for normalization. Defaults to 0.
        normalize_max (float, optional): Maximum value for normalization. Defaults to 1.

    Returns:
        Train and Test DataLoader objects.
    """

    def normalize_data(x, normalize_min, normalize_max):
        """Normalizes data to a range [min, max]."""
        return (x - normalize_min) / (normalize_max - normalize_min)

    mnist_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            (
                torchvision.transforms.Resize(size)
                if isinstance(size, int)
                else torchvision.transforms.Resize(size[::-1])
            ),
            torchvision.transforms.Lambda(
                lambda x: normalize_data(x, normalize_min, normalize_max)
            ),
        ]
    )

    mnist_train = datasets.MNIST(
        root="data/mnist_data",
        train=True,
        download=True,
        transform=mnist_transform,
    )

    mnist_test = datasets.MNIST(
        root="data/mnist_data", train=False, download=True, transform=mnist_transform
    )

    if batch_size is not None:
        train_dataloader = torch.utils.data.DataLoader(
            dataset=mnist_train,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, labels),
        )

        test_dataloader = torch.utils.data.DataLoader(
            dataset=mnist_test,
            batch_size=70000 - batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, labels),
        )

        return train_dataloader, test_dataloader

    return mnist_train, mnist_test


def collate_fn(batch, labels: list):
    """
    Batches the images wrt the provided labels.
    """
    new_batch = []
    for img, label in batch:
        if label in labels:
            new_batch.append((img, label))
    return torch.utils.data.default_collate(new_batch)
