import tensorflow as tf
import darknet
import layers

NUM_CLASSES = 2

def YOLOv3(inputs):

    # Darknet
    route_1, route_2, inputs = darknet.darknet(inputs)


    # Large detections
    inputs = layers.conv_layer(name="conv_00", inputs=inputs, filters=512, kernel_size=1)
    inputs = layers.conv_layer(name="conv_01", inputs=inputs, filters=1024, kernel_size=3)
    inputs = layers.conv_layer(name="conv_02", inputs=inputs, filters=512, kernel_size=1)
    inputs = layers.conv_layer(name="conv_03", inputs=inputs, filters=1024, kernel_size=3)   
    inputs = layers.conv_layer(name="conv_04", inputs=inputs, filters=512, kernel_size=1)
    
    large_detections = layers.conv_layer(name="conv_05", inputs=inputs, filters=1024, kernel_size=3)
    large_detections = layers.conv(name="conv_06", inputs=large_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, batch_normalize=False, activation=False)

    inputs = layers.conv_layer(name="conv_07", inputs=inputs, filters=256, kernel_size=1, batch_norm=True)
    inputs = layers.upsample_layer(name="upsample_0", inputs=inputs)
    
    inputs = layers.route_layer(name="route_0", inputs=inputs, route=route_2)


    # Medium detections
    inputs = layers.conv_layer(name="conv_08", inputs=inputs, filters=256, kernel_size=1)
    inputs = layers.conv_layer(name="conv_09", inputs=inputs, filters=512, kernel_size=3)
    inputs = layers.conv_layer(name="conv_10", inputs=inputs, filters=256, kernel_size=1)
    inputs = layers.conv_layer(name="conv_11", inputs=inputs, filters=512, kernel_size=3)   
    inputs = layers.conv_layer(name="conv_12", inputs=inputs, filters=256, kernel_size=1)

    medium_detections = layers.conv_layer(name="conv_13", inputs=inputs, filters=512, kernel_size=3)
    medium_detections = layers.conv(name="conv_14", inputs=medium_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, batch_normalize=False, activation=False)

    inputs = layers.conv_layer(name="conv_15", inputs=inputs, filters=128, kernel_size=1, batch_norm=True)
    inputs = layers.upsample_layer(name="upsample_1", inputs=inputs)

    inputs = layers.route_layer(name="route_1", inputs=inputs, route=route_1)


    # Small detections
    inputs = layers.conv_layer(name="conv_16", inputs=inputs, filters=128, kernel_size=1)
    inputs = layers.conv_layer(name="conv_17", inputs=inputs, filters=256, kernel_size=3)
    inputs = layers.conv_layer(name="conv_18", inputs=inputs, filters=128, kernel_size=1)
    inputs = layers.conv_layer(name="conv_19", inputs=inputs, filters=256, kernel_size=3)   
    inputs = layers.conv_layer(name="conv_20", inputs=inputs, filters=128, kernel_size=1)

    small_detections = layers.conv_layer(name="conv_21", inputs=inputs, filters=256, kernel_size=3)
    small_detections = layers.conv(name="conv_22", inputs=small_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, batch_normalize=False, activation=False)

    return [small_detections, medium_detections, large_detections]
