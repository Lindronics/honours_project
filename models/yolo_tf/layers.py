import tensorflow as tf

def conv_layer(name, inputs, filters: int, kernel_size: int, downsample: bool=False, batch_normalize: bool=True, activation: bool=True):

    with tf.variable_scope(name):

        params = {
            "inputs": inputs,
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

        inputs = tf.layers.conv2d(**params)
            
        if batch_normalize:
            inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.9, epsilon=1e-05)
        
        if activation:
            inputs = tf.nn.leaky_relu(features=inputs, alpha=0.1)
        
        shortcut = inputs

        return inputs, shortcut


def shortcut_layer(name: str, shortcut, inputs):

    with tf.variable_scope(name):

        inputs += shortcut
        new_shortcut = inputs

    return inputs, new_shortcut


def residual_block(name, inputs, num_filters: int):

    with tf.variable_scope(name):

        inputs, _ = conv_layer(name=name+"_conv_1", inputs=inputs, filters=num_filters, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = conv_layer(name=name+"_conv_2", inputs=inputs, filters=num_filters * 2, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name=name+"_shortcut", shortcut=shortcut, inputs=inputs)

    return inputs, shortcut

def route_layer(name, inputs, route):

    with tf.variable_scope(name):
        inputs = tf.concat([inputs, route], axis=-1)
    
    return inputs


def upsample(x):
    x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method="nearest")
    return x


class YoloBatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)