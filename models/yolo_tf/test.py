import tensorflow as tf

from yolov3 import yolo_v3


input_layer  = tf.keras.layers.Input([412, 412, 3])
feature_maps = yolo_v3(input_layer)

print(feature_maps)
print("FINISHED")