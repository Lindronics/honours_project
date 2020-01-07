import tensorflow as tf

from yolov3 import yolo_v3, decode


input_layer  = tf.keras.layers.Input([412, 412, 3])
feature_maps = yolo_v3(input_layer)

print(feature_maps)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

print(bbox_tensors)

print("FINISHED")