import numpy as np
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from neqr import NEQR


def load_mnist_data():
    '''
    Loads MNNIST dataset from PyTorch using DataLoader.
    :return: Train and Test DataLoader objects.
    '''

    mnist_train = datasets.MNIST(
        root='../FRQI/data',
        train=True,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )

    mnist_test = datasets.MNIST(
        root='../FRQI/data',
        train=False,
        download=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
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


def plot_images(img_dataset):
    '''
    Plots images from a DataLoader object in a 3x3 subplot.
    :param: img_dataset: DataLoader object.
    :return: None
    '''
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_dataset[i][0], cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == '__main__':
    train, test = load_mnist_data()
    train_data = iter(train)
    train_img, train_labels = next(train_data)

    print(train_img.shape, train_labels.shape)

    torch.set_printoptions(precision=0)

    i = 0

    # plt.hist(np.array(train_img[i]).ravel(), bins=50, density=True);
    # plt.xlabel("pixel values")
    # plt.ylabel("relative frequency")
    # plt.title("distribution of pixels")
    # plt.show()

    new_img = torchvision.transforms.Resize(2)(train_img[i])
    normal = torchvision.transforms.Normalize(mean=0., std=(1 / 255.))(new_img)

    new_dimen = normal.reshape(-1, 4)
    new_dimen_list = new_dimen.tolist()

    color_vals = new_dimen_list[0]
    image_size = (2, 2)

    circ = NEQR(image_size, color_vals)
    circ = circ.image_encoding(measure=True)
    circ.decompose().draw('mpl')
    plt.show()

    # Measurement results
    counts = NEQR.get_simulator_result(
        circ,
        backend="qasm_simulator",
        shots=1024,
        plot_counts=True,
    )
    print(counts)
