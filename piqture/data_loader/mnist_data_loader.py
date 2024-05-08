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
    img_size: Union[int, tuple[int, int]] = 28,
    batch_size: int = None,
    labels: list = None,
    normalize_min: float = None,
    normalize_max: float = None,
):
    """
    Loads MNIST dataset from PyTorch using DataLoader.

    Args:
        img_size (int or tuple[int, int], optional): Size to which images
        will be resized. Defaults to 28.
            If integer, images will be resized to a square of that size.
            If tuple, images will be resized to specified height and width.
        batch_size (int, optional): Batch size for the dataset.
        labels (list): List of desired labels.
        normalize_min (float, optional): Minimum value for normalization.
        normalize_max (float, optional): Maximum value for normalization.

    Returns:
        Train and Test DataLoader objects.
    """

    # Check if batch_size is int

    if labels is None:
        labels = list(range(10))
    # Check if labels is list

    if batch_size is None:
        batch_size = 64

    # Check if img_size is int or tuple
    # if not isinstance(img_size, (int, tuple)):
    #     print("some error")
    # if not all((isinstance(dim, int) for dim in dims) for dims in img_size):
    #     raise TypeError("The argument img_size must be of the type int or tuple[int, int]")

    # Check if normalize_min and max are floats, int, pi == Numbers

    mnist_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            (
                torchvision.transforms.Resize(img_size)
                if isinstance(img_size, int)
                else torchvision.transforms.Resize(img_size[::-1])
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


def normalize_data(x, normalize_min, normalize_max):
    """Normalizes data to a range [min, max]."""
    return (x - normalize_min) / (normalize_max - normalize_min)


def collate_fn(batch, labels: list):
    """
    Batches the images wrt the provided labels.
    """
    new_batch = []
    for img, label in batch:
        if label in labels:
            new_batch.append((img, label))
    return torch.utils.data.default_collate(new_batch)
