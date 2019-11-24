import numpy as np

from datetime import datetime

import tensorflow as tf
from tensorflow import nn


class YOLO:

    def __init__(self, config_path, debug=False):
        self.blocks = self.read_cfg(config_path)

        self.x = tf.compat.v1.placeholder('float32', [None, 448, 448, 3])
        layers = [self.x]

        # Net info
        self.input_dimension = int(self.blocks[0]["height"])
        self.channels = int(self.blocks[0]["channels"])

        # Start off with no detections
        detections = []

        for idx, block in enumerate(self.blocks[1:], 1):
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
                anchors = [x.split(",") for x in block["anchors"].split(",  ")]
                config = {
                    "num_classes": int(block["num"]),
                    "anchors": [[int(x) for x in anchor] for anchor in anchors],
                }
                if debug:
                    print(f"Adding yolo layer at {idx}")
                yolo_layer = self.yolo_layer(idx, layers[idx-1], **config)
                layers.append(yolo_layer)
                detections.append(yolo_layer)
        self.layers = layers
        self.detections = tf.concat(detections, 3) # TODO review dimension
        # return self.detections


    def conv_layer(self, idx, x, filters, kernel_size, stride, pad=False, batch_normalize=False, train=False):
        """ Convolutional layer

        Performs a standard 2D convolution on input x.
        """
        # Designate as one convolutional block for graph visualization
        with tf.name_scope(f"{idx}_conv") as scope:

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

            x = nn.leaky_relu(x, name=scope)
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


    def yolo_layer(self, idx, x, num_classes, anchors):
        """ Creates a YOLO layer

        The YOLO layer decodes the convolutional output tensor
        into a tensor containing a list of detections.

        Taken from https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py
        TODO: rewrite this
        """

        with tf.name_scope(f"{idx}_yolo") as scope:

            conv_shape       = tf.shape(x)
            stride           = tf.cast(self.input_dimension // conv_shape[2], tf.float32)
            batch_size       = conv_shape[0]
            output_size      = conv_shape[1]
            anchor_per_scale = len(anchors)

            x = tf.reshape(x, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))

            conv_raw_dxdy = x[:, :, :, :, 0:2]
            conv_raw_dwdh = x[:, :, :, :, 2:4]
            conv_raw_conf = x[:, :, :, :, 4:5]
            conv_raw_prob = x[:, :, :, :, 5: ]

            y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
            x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

            xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
            xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
            xy_grid = tf.cast(xy_grid, tf.float32)

            pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
            pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
            pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

            pred_conf = tf.sigmoid(conv_raw_conf)
            pred_prob = tf.sigmoid(conv_raw_prob)

            return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1, name=scope)


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


tf.compat.v1.disable_eager_execution()



with tf.Session() as sess:
    t = YOLO("cfg/yolov3.cfg")
    # t.network(debug=True)
    writer = tf.summary.FileWriter("logs/func", sess.graph)
    print(sess.run(tf.random.uniform([1, 448, 448, 3])))
    writer.close()