"""
Test Suite for the MNIST Data Loader

This module contains a series of unit tests for the MNIST data loader
implemented in the `mnist_data_loader` module. It verifies that the
data loader functions as expected, checking various aspects such as
loading the dataset, batching of images and labels, normalization,
resizing images, and filtering by labels.

The tests include:
- Loading both training and testing datasets.
- Validating the dimensions of images in batches.
- Ensuring proper normalization of image pixel values.
- Resizing images to the nearest power of 2 dimensions.
- Filtering batches based on specified labels.
- Confirming that DataLoader instances are created successfully.

These tests utilize the pytest framework for structured testing and
assertion checking.
"""

import math

import torch
import torchvision.transforms.functional as F

from piqture.data_loader.mnist_data_loader import load_mnist_dataset
from piqture.transforms import MinMaxNormalization


def test_load_mnist_dataset():
    """
    Test that the dataset loads without errors and returns DataLoader objects
    """
    train_loader, test_loader = load_mnist_dataset(
        batch_size_train=64, batch_size_test=1000, load="both"
    )

    # Check that the loaders are instances of DataLoader
    assert isinstance(
        train_loader, torch.utils.data.DataLoader
    ), "Train loader should be a DataLoader"
    assert isinstance(
        test_loader, torch.utils.data.DataLoader
    ), "Test loader should be a DataLoader"


def test_dataloader_batches():
    """
    Test that the DataLoader batches have the correct image and label shape
    """
    train_loader, test_loader = load_mnist_dataset(
        batch_size_train=64, batch_size_test=1000, load="both"
    )  # pylint: disable=C0301

    # Test a single batch from the train loader
    for images, labels in train_loader:
        assert images.shape[0] == 64, "Train batch size should be 64"
        assert images.shape[1] == 1, "Each image should have 1 channel"
        assert images.shape[2] == 28, "Each image should have height of 28 pixels"
        assert images.shape[3] == 28, "Each image should have width of 28 pixels"
        assert len(labels) == 64, "There should be 64 labels in the batch"
        break  # Only need to check one batch

    # Test a single batch from the test loader
    for images, labels in test_loader:
        assert images.shape[0] == 1000, "Test batch size should be 1000"
        assert images.shape[1] == 1, "Each image should have 1 channel"
        assert images.shape[2] == 28, "Each image should have height of 28 pixels"
        assert images.shape[3] == 28, "Each image should have width of 28 pixels"
        assert len(labels) == 1000, "There should be 1000 labels in the batch"
        break  # Only need to check one batch


def test_resizing_images():
    """
    Test resizing of images.
    """
    train_loader = load_mnist_dataset(load="train", batch_size_train=64)

    for images, _ in train_loader:
        # Assuming you want to retrieve the first image only
        image = images[0]  # pylint: disable=E1136

        assert image.dim() == 3, "Image tensor should have 3 dimensions (C, H, W)"
        height, width = image.squeeze().size()

        # Resize image to the nearest power of 2
        new_height = 2 ** math.ceil(math.log2(height))
        new_width = 2 ** math.ceil(math.log2(width))

        # Resize image using torchvision's functional transforms
        image_resized = F.resize(image, (new_height, new_width))

        # Check that the resized dimensions are correct
        assert (
            image_resized.shape[2] == new_height
        ), f"Image height should be {new_height} after resizing."  # pylint: disable=C0301
        break  # Only need to check one image


def test_normalization_after_resizing():
    """
    Test normalization after resizing.
    """
    train_loader = load_mnist_dataset(load="train", batch_size_train=64)

    for images, _ in train_loader:
        image = images[0]  # pylint: disable=E1136

        # Resize image to the nearest power of 2
        height, width = image.squeeze().size()
        new_height = 2 ** math.ceil(math.log2(height))
        new_width = 2 ** math.ceil(math.log2(width))
        image_resized = F.resize(image, (new_height, new_width))

        # Apply MinMaxNormalization
        normalizer = MinMaxNormalization(normalize_min=0, normalize_max=1)
        image_normalized = normalizer(image_resized)

        # Check normalization
        min_val = image_normalized.min().item()
        max_val = image_normalized.max().item()
        assert (
            0 <= min_val < 1
        ), "Normalized image pixels should be between 0 and 1 (min value)"
        assert (
            0 < max_val <= 1
        ), "Normalized image pixels should be between 0 and 1 (max value)"
        break
