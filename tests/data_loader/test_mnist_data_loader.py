import pytest
import torch
from piqture.data_loader import load_mnist_dataset

# Fixture for loading MNIST dataset
@pytest.fixture
def mnist_data():
    return load_mnist_dataset(
        img_size=28, 
        batch_size_train=64, 
        batch_size_test=1000, 
        normalize_min=0, 
        normalize_max=1, 
        split_ratio=0.8, 
        load="both"
    )

# Test that the dataset loads without errors and returns DataLoader objects
def test_mnist_dataloaders(mnist_data):
    train_loader, test_loader = mnist_data

    assert isinstance(train_loader, torch.utils.data.DataLoader), "Train loader should be a DataLoader"
    assert isinstance(test_loader, torch.utils.data.DataLoader), "Test loader should be a DataLoader"

# Test that the DataLoader batches have the correct image and label shape
def test_dataloader_batches(mnist_data):
    train_loader, test_loader = mnist_data

    # Get the first batch
    for image_batch, label_batch in train_loader:
        assert image_batch.shape[0] == 64, "Train batch size should be 64"
        assert image_batch.shape[2:] == (28, 28), "Each image should have dimensions 28x28"
        assert len(label_batch) == 64, "There should be 64 labels in the batch"
        break

    # Get the first batch from test loader
    for image_batch, label_batch in test_loader:
        assert image_batch.shape[0] == 1000, "Test batch size should be 1000"
        assert image_batch.shape[2:] == (28, 28), "Each test image should have dimensions 28x28"
        assert len(label_batch) == 1000, "There should be 1000 labels in the test batch"
        break

# Test normalization is applied correctly
def test_normalization(mnist_data):
    train_loader, _ = mnist_data

    # Get the first batch and check normalization
    for image_batch, _ in train_loader:
        min_val = image_batch.min().item()
        max_val = image_batch.max().item()
        
        assert 0 <= min_val < 1, "Image pixels should be normalized between 0 and 1 (min value)"
        assert 0 < max_val <= 1, "Image pixels should be normalized between 0 and 1 (max value)"
        break

# Test loading only the train set
def test_load_train_only():
    train_loader = load_mnist_dataset(load="train", batch_size_train=64)

    assert isinstance(train_loader, torch.utils.data.DataLoader), "Train loader should be a DataLoader"
    
    for image_batch, label_batch in train_loader:
        assert len(image_batch) == 64, "Train batch size should be 64"
        break

# Test loading only the test set
def test_load_test_only():
    test_loader = load_mnist_dataset(load="test", batch_size_test=1000)

    assert isinstance(test_loader, torch.utils.data.DataLoader), "Test loader should be a DataLoader"
    
    for image_batch, label_batch in test_loader:
        assert len(image_batch) == 1000, "Test batch size should be 1000"
        break

# Test custom label filtering in collate function
def test_label_filtering():
    labels = [0, 1]  # Only keep images with labels 0 or 1
    train_loader = load_mnist_dataset(load="train", batch_size_train=64, labels=labels)

    for image_batch, label_batch in train_loader:
        assert all(label in labels for label in label_batch), "All labels should be in the specified label list"
        break
