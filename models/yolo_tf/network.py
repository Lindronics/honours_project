import numpy as np

from datetime import datetime

import tensorflow as tf
from tensorflow import nn


class YOLO:

    def read_cfg(self, fname: str):
        """ Reads and parses yolo v3 config file """

        with open(fname, "r") as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines if x.strip() and x[0] != "#"]

        
        current_block = {}
        blocks = []
        
        for line in lines:
            if line[0] == "[":       
                if len(current_block) != 0:
                    blocks.append(current_block)
                    current_block = {}
                current_block["type"] = line[1:-1].strip()     
            else:
                key, value = line.split("=") 
                current_block[key.strip()] = value.strip()

        blocks.append(current_block)
        return blocks


    # def network(self, debug=False):
    #     self.x = tf.compat.v1.placeholder('float32', [None, 448, 448, 3])
    #     layers = [self.x]
    #     layers.append(self.conv_layer(1, layers[0], filters=32, kernel_size=1, stride=1, pad=True))
    #     layers.append(self.conv_layer(2, layers[1], filters=32, kernel_size=1, stride=1, pad=True))
    #     layers.append(self.shortcut(3, layers[2], layers[1]))
    #     layers.append(self.conv_layer(4, layers[3], filters=32, kernel_size=1, stride=1, pad=True))
    #     layers.append(self.conv_layer(5, layers[4], filters=32, kernel_size=1, stride=1, pad=True))
    #     layers.append(self.route(6, layers[4], layers[5]))
    #     layers.append(self.conv_layer(7, layers[6], filters=32, kernel_size=1, stride=1, pad=True))
    #     layers.append(self.conv_layer(8, layers[7], filters=32, kernel_size=1, stride=1, pad=True))
    #     layers.append(self.conv_layer(9, layers[8], filters=32, kernel_size=1, stride=1, pad=True))
    #     layers.append(self.conv_layer(10, layers[9], filters=32, kernel_size=1, stride=1, pad=True))
    #     layers.append(self.conv_layer(11, layers[10], filters=32, kernel_size=1, stride=1, pad=True))
    #     self.layers = layers
    #     return self.layers


    def network(self, debug=False):
        self.x = tf.compat.v1.placeholder('float32', [None, 448, 448, 3])
        layers = [self.x]
        blocks = self.read_cfg("cfg/yolov3.cfg")

        for idx, block in enumerate(blocks[1:7], 1):
            if block["type"] == "convolutional":
                config = {
                    "batch_normalize": False, # TODO fix this
                    # "batch_normalize": block["batch_normalize"] == "1", 
                    "filters": int(block["filters"]),
                    "kernel_size": int(block["size"]),
                    "stride": int(block["stride"]),
                    "pad": block["pad"] == "1",
                }
                if debug:
                    print(f"Adding conv layer at {idx}")
                layers.append(self.conv_layer(idx, layers[idx-1], **config))

            elif block["type"] == "upsample":
                layers.append(self.upsample_layer(idx, layers[idx-1], stride=int(block["stride"])))

            elif block["type"] == "route":
                # Convert relative to absolute index
                route_layers = [int(x) for x in block["layers"].split(",")]
                route_layers = [idx+x for x in route_layers if x < 0]

                if debug:
                    print(f"Adding route layer at {idx}")

                if len(route_layers) == 2:
                    layers.append(self.route(idx, route_layers))
                else:
                    layers.append(self.route(idx, layers[route_layers[0]]))

            elif block["type"] == "shortcut":
                # Shortcut index is always negative in config
                if debug:
                    print(f"Adding shortcut layer at {idx} from {idx-1} to {idx+int(block['from'])}")
                layers.append(self.shortcut(idx, layers[idx-1], layers[idx+int(block["from"])]))

            elif block["type"] == "yolo":
                if debug:
                    print(f"Adding yolo layer at {idx}")
                layers.append([])
        self.layers = layers
        return self.layers


    def conv_layer(self, idx, x, filters, kernel_size, stride, pad=False, batch_normalize=False, train=False):
        """ Convolutional layer

        Performs a standard 2D convolution on input x.
        """

        # Randomly initialize filter weights
        channels = x.get_shape()[3]
        filter_weights = tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, channels, filters], stddev=0.1), trainable=train, name=f"{idx}_conv_weights")

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
            biases = tf.Variable(tf.constant(0.1, shape=[filters]), trainable=train, name=f"{idx}_conv_bias")
            x = tf.add(x, biases, name=f"{idx}_conv_biased")

        x = nn.leaky_relu(x)
        return x


    def upsample_layer(self, idx, x, stride):
        """ Upsampling layer

        Upsamples the input x by factor of stride
        """
        new_size = tf.constant(x.shape.dims[1:3], dtype="int32") // stride
        x = tf.image.resize(x, size=new_size, method="bilinear", name=f"{idx}_upsample")
        return x


    def route(self, idx, a, b=None):
        """ Route layer

        Concatenates feature maps of two layers a and b
        """
        
        # If end is defined, return concatenated feature maps
        if b!=None:
            return tf.concat([a, b], axis=3, name=f"{idx}_route") # TODO verify dimension (channel)

        # Else, just return output of start
        return a


    def shortcut(self, idx, a, b):
        """ Shortcut layer

        Adds the feature maps of a and b
        """
        return tf.add(a, b, name=f"{idx}_shortcut")


tf.compat.v1.disable_eager_execution()



with tf.Session() as sess:
    t = YOLO()
    t.network(debug=True)
    writer = tf.summary.FileWriter("logs/func", sess.graph)
    # print(sess.run(tf.random.uniform([None, 448, 448, 3])))
    writer.close()