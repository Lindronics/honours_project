import tensorflow as tf
import darknet
import layers

NUM_CLASSES = 2

def YOLOv3(inputs):

    # Darknet
    route_1, route_2, inputs = darknet.darknet(inputs)


    # Large detections
    inputs, _ = layers.conv_layer(name='conv_52', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_53', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_54', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_55', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_normalize=True, activation=True)   
    inputs, _ = layers.conv_layer(name='conv_56', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_normalize=True, activation=True)
    
    large_detections, _ = layers.conv_layer(name='conv_57', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_normalize=True, activation=True)
    large_detections, _ = layers.conv(name='conv_58', inputs=large_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, downsample=False, batch_normalize=False, activation=False)

    inputs, _ = layers.conv_layer(name='conv_59', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation=True)
    inputs = layers.upsample_layer(name='upsample_0', inputs=inputs)
    
    inputs = layers.route_layer(name='route_0', inputs=inputs, route=route_2)


    # Medium detections
    inputs, _ = layers.conv_layer(name='conv_52', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_53', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_54', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_55', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_normalize=True, activation=True)   
    inputs, _ = layers.conv_layer(name='conv_56', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_normalize=True, activation=True)

    medium_detections, _ = layers.conv_layer(name='conv_57', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_normalize=True, activation=True)
    medium_detections, _ = layers.conv(name='conv_58', inputs=medium_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, downsample=False, batch_normalize=False, activation=False)

    inputs, _ = layers.conv_layer(name='conv_59', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation=True)
    inputs = layers.upsample_layer(name='upsample_0', inputs=inputs)

    inputs = layers.route_layer(name='route_0', inputs=inputs, route=route_1)


    # Small detections
    inputs, _ = layers.conv_layer(name='conv_52', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_53', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_54', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_normalize=True, activation=True)
    inputs, _ = layers.conv_layer(name='conv_55', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_normalize=True, activation=True)   
    inputs, _ = layers.conv_layer(name='conv_56', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_normalize=True, activation=True)

    small_detections, _ = layers.conv_layer(name='conv_57', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_normalize=True, activation=True)
    small_detections, _ = layers.conv(name='conv_58', inputs=small_detections, filters=3*(NUM_CLASSES+5), kernel_size=1, downsample=False, batch_normalize=False, activation=False)

    return [small_detections, medium_detections, large_detections]
