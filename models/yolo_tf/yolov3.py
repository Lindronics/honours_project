# Contains code by YunYang1994

import tensorflow as tf
import numpy as np
import json

import darknet
import layers

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)

# Load configuration
with open("config.json", "r") as f:
    CONFIG = json.load(f)

ANCHORS = get_anchors("anchors.txt")
NUM_CLASSES = CONFIG["NETWORK"]["NUM_CLASSES"]
STRIDES = CONFIG["NETWORK"]["STRIDES"]
IOU_LOSS_THRESH = CONFIG["TRAINING"]["IOU_LOSS_THRESHOLD"]

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

def bbox_iou(boxes1, boxes2):
    """ From https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3 """

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):
    """ 
    From https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3 
    
    Parameters
    ----------
    boxes1: tensor
        Shape:
        [batch_size, layer_height, layer_width, channels, 4]
        4 stems from: x_center, y_center, width, height
    boxes2: tensor
        Shape:
        [batch_size, layer_height, layer_width, channels, 4]
        4 stems from: x_center, y_center, width, height
    """

    # Convert bboxes from [x_center, y_center, w, h] to [x_lu, y_lu, x_rd, y_rd]
    boxes1 = [boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5]
    boxes2 = [boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5]
    boxes1 = tf.concat(boxes1, axis=-1)
    boxes2 = tf.concat(boxes2, axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    # Compute area
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Find enclosing area of both bboxes
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # Compute intersection, union, and IOU
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    # Compute GIOU
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):
    """ From https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3 """

    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASSES))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    if tf.math.is_nan(giou_loss):
        print()

    return giou_loss, conf_loss, prob_loss