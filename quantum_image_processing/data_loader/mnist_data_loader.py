import torch.utils.data
import torchvision
import torchvision.datasets as datasets


def load_mnist_data():
    """
    Loads MNIST dataset from PyTorch using DataLoader.
    :return: Train and Test DataLoader objects.
    """

    mnist_train = datasets.MNIST(
        root="../data_loader/mnist_data",
        train=True,
        download=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(2)
        ])
    )

    mnist_test = datasets.MNIST(
        root="../data_loader/mnist_data",
        train=False,
        download=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(2)
        ])
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=mnist_train,
        batch_size=1,
        shuffle=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=mnist_test,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, test_dataloader
