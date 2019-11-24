import numpy as np

import tensorflow as tf
from tensorflow import nn

# from loader import read_cfg


class YOLO:

    def network(self):
        self.x = tf.Variable(tf.zeros(shape=[1, 448, 448, 3]))
        self.conv1 = self.conv_layer(1, self.x, filters=32, kernel_size=1, stride=1, pad=True)
        self.conv2 = self.conv_layer(2, self.conv1, filters=64, kernel_size=3, stride=2, pad=True)
        self.ups1 = self.upsample_layer(3, self.conv2, stride=2)

    def conv_layer(self, idx, x, filters, kernel_size, stride, pad=False, batch_normalize=False, train=False):

        # Randomly initialize filter weights
        channels = x.get_shape()[3]
        filter_weights = tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, channels, filters], stddev=0.1), trainable=train)

        # Pad tensor if necessary
        if pad:
            padding_size = kernel_size // 2
            padding = tf.constant([[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
            x = tf.pad(x, padding)

        # Convolutional layer
        x =  nn.conv2d(x, filter_weights, [1, stride, stride, 1], padding="VALID")

        # Normalization or bias
        if batch_normalize:
            x = nn.batch_normalization(x, name=f"{idx}_conv_normalized")
        else:
            biases = tf.Variable(tf.constant(0.1, shape=[filters]), trainable=train)
            x = tf.add(x, biases, name=f"{idx}_conv_biased")

        x = nn.leaky_relu(x)
        return x

    def upsample_layer(self, idx, x, stride):
        new_size = tf.constant(x.get_shape()[1:3], dtype="int32") // stride
        x = tf.image.resize(x, size=new_size, method="nearest", name=f"{idx}_upsample")
        return x

t = YOLO()
t.network()