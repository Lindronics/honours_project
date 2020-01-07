# Contains code by YunYang1994

import tensorflow as tf
import numpy as np

import darknet
import layers

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)

NUM_CLASSES = 2
ANCHORS = get_anchors("anchors.txt")
STRIDES = [8, 16, 32]

def yolo_v3(inputs):

    # Darknet
    route_1, route_2, inputs = darknet.darknet(inputs)


    # Large detections
    inputs = layers.conv_layer(name="conv_00", inputs=inputs, filters=512, kernel_size=1)
    inputs = layers.conv_layer(name="conv_01", inputs=inputs, filters=1024, kernel_size=3)
    inputs = layers.conv_layer(name="conv_02", inputs=inputs, filters=512, kernel_size=1)
    inputs = layers.conv_layer(name="conv_03", inputs=inputs, filters=1024, kernel_size=3)   
    inputs = layers.conv_layer(name="conv_04", inputs=inputs, filters=512, kernel_size=1)
    
    large_detections = layers.conv_layer(name="conv_05", inputs=inputs, filters=1024, kernel_size=3)
    large_detections = layers.conv_layer(name="conv_06", inputs=large_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, batch_normalize=False, activation=False)

    inputs = layers.conv_layer(name="conv_07", inputs=inputs, filters=256, kernel_size=1)
    inputs = layers.upsample_layer(name="upsample_0", inputs=inputs)
    
    inputs = layers.route_layer(name="route_0", inputs=inputs, route=route_2)


    # Medium detections
    inputs = layers.conv_layer(name="conv_08", inputs=inputs, filters=256, kernel_size=1)
    inputs = layers.conv_layer(name="conv_09", inputs=inputs, filters=512, kernel_size=3)
    inputs = layers.conv_layer(name="conv_10", inputs=inputs, filters=256, kernel_size=1)
    inputs = layers.conv_layer(name="conv_11", inputs=inputs, filters=512, kernel_size=3)   
    inputs = layers.conv_layer(name="conv_12", inputs=inputs, filters=256, kernel_size=1)

    medium_detections = layers.conv_layer(name="conv_13", inputs=inputs, filters=512, kernel_size=3)
    medium_detections = layers.conv_layer(name="conv_14", inputs=medium_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, batch_normalize=False, activation=False)

    inputs = layers.conv_layer(name="conv_15", inputs=inputs, filters=128, kernel_size=1)
    inputs = layers.upsample_layer(name="upsample_1", inputs=inputs)

    inputs = layers.route_layer(name="route_1", inputs=inputs, route=route_1)


    # Small detections
    inputs = layers.conv_layer(name="conv_16", inputs=inputs, filters=128, kernel_size=1)
    inputs = layers.conv_layer(name="conv_17", inputs=inputs, filters=256, kernel_size=3)
    inputs = layers.conv_layer(name="conv_18", inputs=inputs, filters=128, kernel_size=1)
    inputs = layers.conv_layer(name="conv_19", inputs=inputs, filters=256, kernel_size=3)   
    inputs = layers.conv_layer(name="conv_20", inputs=inputs, filters=128, kernel_size=1)

    small_detections = layers.conv_layer(name="conv_21", inputs=inputs, filters=256, kernel_size=3)
    small_detections = layers.conv_layer(name="conv_22", inputs=small_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, batch_normalize=False, activation=False)

    return [small_detections, medium_detections, large_detections]


def decode_predictions(raw_predictions, i):

    batch_size       = tf.shape(raw_predictions)[0]
    output_size      = tf.shape(raw_predictions)[1]

    raw_predictions = tf.reshape(raw_predictions, (batch_size, output_size, output_size, 3, 5 + NUM_CLASSES))

    conv_raw_xy             = raw_predictions[... , 0:2]
    conv_raw_wh             = raw_predictions[... , 2:4]
    conv_raw_confidence     = raw_predictions[... , 4:5]
    conv_raw_probabilities  = raw_predictions[... , 5: ]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, None], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[None, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, None], y[:, :, None]], axis=-1)
    xy_grid = tf.tile(xy_grid[None, :, :, None, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_xy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_wh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_confidence)
    pred_prob = tf.sigmoid(conv_raw_probabilities)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

