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

from functools import partial
from typing import Union

import torch.utils.data
import torchvision
from torchvision import datasets

from piqture.transforms import MinMaxNormalization


def load_mnist_dataset(
    img_size: Union[int, tuple[int, int]] = 28,
    batch_size: int = None,
    labels: list = None,
    normalize_min: float = None,
    normalize_max: float = None,
    train_test: int = None,
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
    # Check if img_size is int or tuple.
    # Also check if tuple entries are int.
    if not isinstance(img_size, (int, tuple)):
        raise TypeError(
            "the input img_size must be of the type int or tuple[int, int]."
        )

    if isinstance(img_size, tuple) and not all(
        isinstance(size, int) for size in img_size
    ):
        raise TypeError("the input img_size must be of the type tuple[int, int].")

    # Check if batch_size is an int.
    if batch_size:
        if not isinstance(batch_size, int):
            raise TypeError("The input batch_size must be of the type int.")

    # Check if labels are a list.
    if labels:
        if not isinstance(labels, list):
            raise TypeError("The input labels must be of the type list.")

    if normalize_max and normalize_min:
        # Define a custom mnist transforms.
        mnist_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(img_size),
                MinMaxNormalization(normalize_min, normalize_max),
            ]
        )
    else:
        # When normalization is not requested; when normalize_min and max are None.
        mnist_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(img_size),
            ]
        )

    # Partial application of custom collate function that calls
    # collate_fn with args.
    new_batch = []
    custom_collate = partial(collate_fn, labels=labels, new_batch=new_batch)

    # Download dataset.
    mnist_train = datasets.MNIST(
        root="data/mnist_data",
        train=True,
        download=True,
        transform=mnist_transform,
    )

    mnist_test = datasets.MNIST(
        root="data/mnist_data", train=False, download=True, transform=mnist_transform
    )

    if labels or batch_size:
        train_dataloader = torch.utils.data.DataLoader(
            dataset=mnist_train,
            batch_size=batch_size if batch_size is not None else 1,
            shuffle=False,
            collate_fn=custom_collate,
        )

        test_dataloader = torch.utils.data.DataLoader(
            dataset=mnist_test,
            batch_size=70000 - batch_size if batch_size is not None else 1,
            shuffle=False,
            collate_fn=custom_collate,
        )

        return train_dataloader, test_dataloader
    # 1 for test
    # 2 for train
    #  rest for both
    if (train_test==1):
        return mnist_test
    elif (train_test==2):
        return mnist_train
    else :
        return mnist_train, mnist_test


def collate_fn(batch, labels: list, new_batch: list):
    """
    Batches the images wrt the provided labels.
    """
    for img, label in batch:
        if label in labels:
            new_batch.append((img, label))
    # If new batch is empty return empty list.
    if len(new_batch) > 0:
        return torch.utils.data.default_collate(new_batch)
    return []
