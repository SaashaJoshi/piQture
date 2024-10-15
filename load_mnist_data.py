import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

def load_testtrain_mnist (a):
    # a is deciding factor  0 for train and else for test
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
    if a==0:
        return trainX,trainY
    else :
        return testX ,testY
