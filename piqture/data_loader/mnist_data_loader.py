"""Data Loader for MNIST images"""

from __future__ import annotations

from functools import partial
from typing import Union, Tuple

import torch.utils.data
import torchvision
from torchvision import datasets

from piqture.transforms import MinMaxNormalization


def load_mnist_dataset(
    img_size: Union[int, Tuple[int, int]] = 28,
    batch_size_train: int = 64,
    batch_size_test: int = 1000,
    labels: list = None,
    normalize_min: float = None,
    normalize_max: float = None,
    split_ratio: float = 0.8,
    load: str = "both",  # Options: "train", "test", or "both"
):
    """
    Loads MNIST dataset from PyTorch, optionally splits and returns DataLoader objects.

    Args:
        img_size (int or tuple[int, int], optional): Size to which images
            will be resized. Defaults to 28.
        batch_size_train (int, optional): Batch size for training set. Defaults to 64.
        batch_size_test (int, optional): Batch size for test set. Defaults to 1000.
        labels (list, optional): List of desired labels.
        normalize_min (float, optional): Minimum value for normalization.
        normalize_max (float, optional): Maximum value for normalization.
        split_ratio (float, optional): Ratio to split train/test datasets. Defaults to 0.8.
        load (str, optional): Indicates whether to load "train", "test", or "both". Defaults to "both".

    Returns:
        Train and/or Test DataLoader objects, depending on the `load` argument.
    """

    if not isinstance(img_size, (int, tuple)):
        raise TypeError("img_size must be an int or tuple[int, int].")

    if isinstance(img_size, tuple) and not all(isinstance(size, int) for size in img_size):
        raise TypeError("img_size tuple must contain integers.")

    if not isinstance(batch_size_train, int) or not isinstance(batch_size_test, int):
        raise TypeError("batch_size_train and batch_size_test must be integers.")

    if labels and not isinstance(labels, list):
        raise TypeError("labels must be a list.")

    if load not in {"train", "test", "both"}:
        raise ValueError('load must be one of "train", "test", or "both".')

    # Define the transform
    mnist_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_size),
        MinMaxNormalization(normalize_min, normalize_max) if normalize_min and normalize_max else torchvision.transforms.Lambda(lambda x: x)
    ])

    # Load the full MNIST dataset
    mnist_full = datasets.MNIST(
        root="data/mnist_data",
        train=True,  # Always load train to split later
        download=True,
        transform=mnist_transform
    )

    # Split the dataset into train/test based on split_ratio
    train_size = int(len(mnist_full) * split_ratio)
    test_size = len(mnist_full) - train_size

    mnist_train, mnist_test = torch.utils.data.random_split(
        mnist_full, [train_size, test_size]
    )

    custom_collate = None
    if labels:
        custom_collate = partial(collate_fn, labels=labels, new_batch=[])

    # Prepare dataloaders
    def create_dataloader(dataset, batch_size, collate_fn=None):
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    train_dataloader = create_dataloader(mnist_train, batch_size_train, custom_collate)
    test_dataloader = create_dataloader(mnist_test, batch_size_test, custom_collate)

    if load == "train":
        return train_dataloader
    elif load == "test":
        return test_dataloader
    else:
        return train_dataloader, test_dataloader


def collate_fn(batch, labels: list, new_batch: list):
    """
    Custom collate function that filters batches by provided labels.
    """
    for img, label in batch:
        if label in labels:
            new_batch.append((img, label))
    
    if new_batch:
        return torch.utils.data.default_collate(new_batch)
    return []  # Return empty batch if no matching labels
