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

import pytest
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

from piqture.data_loader.mnist_data_loader import collate_fn, load_mnist_dataset
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


def test_invalid_img_size_type():
    """
    Test that a TypeError is raised when img_size is not an int or tuple of ints.
    """
    with pytest.raises(
        TypeError, match="img_size must be an int or tuple\\[int, int\\]."
    ):
        load_mnist_dataset(img_size="invalid")


def test_invalid_img_size_tuple():
    """
    Test that a TypeError is raised when img_size tuple contains non-integer values.
    """
    with pytest.raises(TypeError, match="img_size tuple must contain integers."):
        load_mnist_dataset(img_size=(28, "invalid"))


def test_invalid_batch_size():
    """
    Test that a TypeError is raised when batch_size_train or batch_size_test is not an int.
    """
    with pytest.raises(
        TypeError, match="batch_size_train and batch_size_test must be integers."
    ):
        load_mnist_dataset(batch_size_train="invalid", batch_size_test=1000)


def test_invalid_labels_type():
    """
    Test that a TypeError is raised when labels is not a list.
    """
    with pytest.raises(TypeError, match="labels must be a list."):
        load_mnist_dataset(labels="invalid")


def test_invalid_load_value():
    """
    Test that a ValueError is raised when load is not 'train', 'test', or 'both'.
    """
    with pytest.raises(
        ValueError, match='load must be one of "train", "test", or "both".'
    ):
        load_mnist_dataset(load="invalid")


def test_valid_load_mnist():
    """
    Test that the dataset loads without errors when passing valid arguments.
    """
    train_loader, test_loader = load_mnist_dataset(
        img_size=28, batch_size_train=64, batch_size_test=1000, load="both"
    )
    assert train_loader, "Train loader should be created."
    assert test_loader, "Test loader should be created."


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


def test_load_mnist_dataset_returns_test_dataloader():
    """
    Test that load_mnist_dataset returns a valid test DataLoader when 'test' is selected.
    """

    # Call the function to load only the test DataLoader
    test_dataloader = load_mnist_dataset(
        img_size=28,
        batch_size_train=64,
        batch_size_test=1000,
        labels=None,
        normalize_min=None,
        normalize_max=None,
        split_ratio=0.8,
        load="test",
    )

    # Check that the returned object is a DataLoader
    assert isinstance(
        test_dataloader, DataLoader
    ), "Returned object is not a DataLoader"

    # Check that the DataLoader has the correct batch size
    assert (
        test_dataloader.batch_size == 1000
    ), "Test DataLoader has incorrect batch size"

    # Fetch a batch of data and check that it returns correctly sized data
    data_iter = iter(test_dataloader)
    images, labels = next(data_iter)

    # Check that the batch has the correct shape: [batch_size, 1, 28, 28]
    assert images.shape == (
        1000,
        1,
        28,
        28,
    ), f"Incorrect image batch shape: {images.shape}"

    # Check that labels are present and have the correct batch size
    assert labels.shape[0] == 1000, f"Incorrect labels batch size: {labels.shape[0]}"

    print("Test DataLoader successfully loaded and verified.")


def test_collate_fn_with_labels():
    """
    Test that collate_fn filters images with specified labels.
    """
    # Create a dummy batch with labels 0, 1, 2, 3
    batch = [
        (torch.rand(1, 28, 28), torch.tensor(0)),
        (torch.rand(1, 28, 28), torch.tensor(1)),
        (torch.rand(1, 28, 28), torch.tensor(2)),
        (torch.rand(1, 28, 28), torch.tensor(3)),
    ]

    # Filter for labels 0 and 1
    filtered_batch = collate_fn(batch, labels=[0, 1])

    # Extract filtered labels, assuming they're returned as a tensor
    _, filtered_labels_tensor = filtered_batch  # pylint: disable=W0632

    # Convert filtered labels tensor to a list of Python integers
    filtered_labels = filtered_labels_tensor.tolist()

    # Check that only images with labels 0 and 1 are returned
    assert filtered_labels == [
        0,
        1,
    ], "Filtered batch should only contain labels 0 and 1."
