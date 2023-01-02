import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'X train: {x_train.shape}')
print(f'Y train: {y_train.shape}')
print(f'X test {x_test.shape}')

# Plot image data
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     # plt.subplot(330 + 1 + i)
#     plt.imshow(x_train[i], cmap=plt.get_cmap('binary_r'))     # default cmap=virdis
# # plt.show()

# resize images
x_train = np.expand_dims(x_train, axis=-1)
resize_x_train = tf.image.resize(x_train, (4, 4))
print(x_train.shape)
print(resize_x_train.shape)

# Plot original and resized images
row = 3
cols = 3
fig, axs = plt.subplots(row, 2*cols)
for i in range(row):
    for j in range(0, 2*cols, 2):
        axs[i, j].imshow(x_train[i+j])
        axs[i, j+1].imshow(resize_x_train[i+j])
    # plt.imshow(resize_x_train[i], cmap=plt.get_cmap('binary_r'))
plt.show()