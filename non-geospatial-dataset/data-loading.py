import torchvision
from torch.utils.data import DataLoader
from torchgeo.datasets import RESISC45
import matplotlib.pyplot as plt

def load_resisc45_data():
    '''
    Loads RESISC45 dataset from PyTorch using DataLoader.
    :return: Train and Test DataLoader objects.
    '''

    # Had issues in extracting RAR files on macOS.
    # Prefer extraction using The UnArchiver application.
    train_data = RESISC45(root='./resisc45_data', download=False, split='train', transforms=None)
    test_data = RESISC45(root='./resisc45_data', download=False, split='test', transforms=None)

    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=10, shuffle=False)

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
    train, test = load_resisc45_data()
    print(len(train), len(test))

    train_data = iter(train)
    next_train_data = next(train_data)
    train_sample, train_label = next_train_data['image'], next_train_data['label']
    print(train_sample.shape, train_label.shape)

    plot_images(train_sample)

