import tensorflow as tf


def conv_layer(name, inputs, filters: int, kernel_size: int, downsample: bool=False, batch_normalize: bool=True, activation: bool=True):

    with tf.compat.v1.variable_scope(name):

        params = {
            "filters": filters,
            "kernel_size": kernel_size,
            "use_bias": not batch_normalize,
        }

        if downsample:
            paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            inputs = tf.pad(inputs, paddings, "CONSTANT")
            params["strides"] = (2, 2)
            params["padding"] = "VALID"
        else:
            params["strides"] = (1, 1)
            params["padding"] = "SAME"

        inputs = tf.keras.layers.Conv2D(**params)(inputs)
            
        if batch_normalize:
            inputs = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(inputs)
        
        if activation:
            inputs = tf.nn.leaky_relu(features=inputs, alpha=0.1)

        return inputs


def shortcut_layer(name: str, shortcut, inputs):

    with tf.compat.v1.variable_scope(name):

        inputs += shortcut
        new_shortcut = inputs

    return inputs, new_shortcut


def residual_block(name, inputs, num_filters: int):

    with tf.compat.v1.variable_scope(name):

        conv = conv_layer(name=name+"_conv_1", inputs=inputs, filters=num_filters, kernel_size=1, downsample=False, batch_normalize=True, activation='LEAKY')
        conv = conv_layer(name=name+"_conv_2", inputs=inputs, filters=num_filters * 2, kernel_size=3, downsample=False, batch_normalize=True, activation='LEAKY')
        
        inputs = inputs + conv

    return inputs


def route_layer(name, inputs, route):

    with tf.compat.v1.variable_scope(name):
        inputs = tf.concat([inputs, route], axis=-1)
    
    return inputs


def upsample_layer(name, inputs):

    with tf.compat.v1.variable_scope(name):
        inputs = tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method="nearest")
    
    return inputs
