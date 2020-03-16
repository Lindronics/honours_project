import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import os

class AbstractModel():
    """ 
    Abstract model class. Provides an interface for all models.
    """

    modes = ["rgb", "lwir", "grayscale", "stacked", "voting", "fusion"]

    def __init__(self, mode, num_classes, input_shape=None, weight_dir=None):
        """
        Params
        ------
        mode: str
            Multispectral mode of the model. 
            Can be rgb, lwir, grayscale, stacked, voting or fusion.
        num_classes: int
            Number of classes
        input_shape: tuple
            Shape of input tensor
        weight_dir: str
            Directory containing weight files
        """
        modes = {
            "rgb": self.rgb,
            "lwir": self.lwir,
            "grayscale": self.grayscale,
            "stacked": self.stacked,
            "voting": self.voting,
            "fusion": self.fusion
        }
        self.method = modes[mode]
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weight_dir = weight_dir
        
    def __call__(self, x):
        return self.method(x)

    def get_model(self):
        # Fix random state
        tf.random.set_seed(42)
        np.random.seed(42)
        
        input_tensor = K.layers.Input(self.input_shape)
        output_tensor = self.method(input_tensor)
        model = K.Model(input_tensor, output_tensor)
        return model

    def net(self, x, fc=True):
        """
        Defines convolutional network.
        """
        raise NotImplementedError

    def fc(self, x):
        """
        Defines fully connected part of network.
        """
        raise NotImplementedError

    def rgb(self, x):
        x = x[:, :, :, 0:3]
        x = self.net(x)
        return x

    def lwir(self, x):
        x = x[:, :, :, 3, None]
        x = self.net(x)
        return x

    def grayscale(self, x):
        x = x[:, :, :, 0:3]
        x = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        x = self.net(x)
        return x

    def stacked(self, x):
        x = self.net(x)
        return x

    def voting(self, x):
        rgb = self.net(x[:, :, :, 0:3])
        lwir = self.net(x[:, :, :, 3, None])
        x = K.layers.Add()([rgb, lwir]) * 0.5
        return x

    def fusion(self, x):
        rgb = self.net(x[:, :, :, 0:3], fc=False)
        lwir = self.net(x[:, :, :, 3, None], fc=False)
        x = K.layers.Concatenate()([rgb, lwir])
        x = self.fc(x)
        return x


class AlexNet(AbstractModel):
    """
    AlexNet implementation with some modifications.
    """

    def net(self, x, fc=True):
        x = K.layers.Conv2D(filters=96, kernel_size=11, strides=2, padding="same", activation="relu")(x)
        x = K.layers.MaxPool2D(pool_size=3, strides=2)(x)

        x = K.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="same", activation="relu")(x)
        x = K.layers.MaxPool2D(pool_size=3, strides=2)(x)

        x = K.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same", activation="relu")(x)
        x = K.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same", activation="relu")(x)
        x = K.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
        x = K.layers.MaxPool2D(pool_size=3, strides=2)(x)

        x = K.layers.Flatten()(x)

        if fc:
            return self.fc(x)
        return x

    def fc(self, x):
        x = K.layers.Dense(1024, activation="relu")(x)
        x = K.layers.Dense(1024, activation="relu")(x)
        x = K.layers.Dense(self.num_classes, activation="softmax")(x)
        return x


class ResNet(AbstractModel):
    """
    Custom network with residual blocks.
    """

    def residual_block(self, x, kernel_size):
        """
        Custom residual block implementation
        """
        x_shortcut = x
        x = K.layers.Conv2D(filters=x.shape[-1], kernel_size=kernel_size, strides=1, padding="same")(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.ReLU()(x)
        x = K.layers.Conv2D(filters=x.shape[-1], kernel_size=kernel_size, strides=1, padding="same")(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Add()([x_shortcut, x])
        x = K.layers.ReLU()(x)
        return x

    def net(self, x, fc=True):
        x = K.layers.Conv2D(filters=32, kernel_size=11, strides=2, activation="relu", padding="valid")(x)
        x = K.layers.MaxPool2D(pool_size=2)(x)

        x = self.residual_block(x, kernel_size=5)
        x = self.residual_block(x, kernel_size=5)
        x = self.residual_block(x, kernel_size=5)

        x = K.layers.Conv2D(filters=64, kernel_size=7, strides=2, activation="relu", padding="valid")(x)
        x = K.layers.MaxPool2D(pool_size=2)(x)

        x = self.residual_block(x, kernel_size=5)
        x = self.residual_block(x, kernel_size=5)
        x = self.residual_block(x, kernel_size=5)

        x = K.layers.Flatten()(x)

        if fc:
            return self.fc(x)
        return x
        
    def fc(self, x):
        x = K.layers.Dense(1024, activation="relu")(x)
        x = K.layers.Dropout(0.25)(x)
        x = K.layers.Dense(512, activation="relu")(x)
        x = K.layers.Dropout(0.25)(x)
        x = K.layers.Dense(self.num_classes, activation="softmax")(x)
        return x


class CustomNet(AbstractModel):
    """
    Custom-built CNN
    """

    def net(self, x, fc=True):
        x = K.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same")(x)
        x = K.layers.Conv2D(filters=16, kernel_size=11, strides=2, padding="valid")(x)
        x = K.layers.LeakyReLU()(x)
        x = K.layers.MaxPool2D(pool_size=11, strides=2)(x)
        x = K.layers.Dropout(0.4)(x)

        x = K.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(x)
        x = K.layers.Conv2D(filters=32, kernel_size=7, strides=2, padding="valid")(x)
        x = K.layers.LeakyReLU()(x)
        x = K.layers.MaxPool2D(pool_size=5, strides=2)(x)
        x = K.layers.Dropout(0.4)(x)

        if fc:
            return self.fc(x)
        return x

    def fc(self, x):
        x = K.layers.Flatten()(x)
        x = K.layers.Dense(64, activation="relu")(x)
        x = K.layers.Dropout(0.4)(x)
        x = K.layers.Dense(self.num_classes, activation="softmax")(x)
        return x


class ResNet152v2(AbstractModel):
    """
    Pre-built ResNet 152 V2 from Keras.
    """

    def net(self, x, fc=True):
        if x.shape[-1] == 3:
            if self.weight_dir:
                tf.print("==> Loading ResNet 152 v2 with imagenet weights for shape", x.shape)
                net = K.applications.resnet_v2.ResNet152V2(include_top=False, weights="imagenet", input_shape=x.shape[1:])
            else:
                tf.print("==> Loading ResNet 152 v2 without weights for shape", x.shape)
                net = K.applications.resnet_v2.ResNet152V2(include_top=False, weights=None, input_shape=x.shape[1:])
        else:
            tf.print("==> Loading ResNet 50 v2 for shape", x.shape)
            net = K.applications.resnet_v2.ResNet50V2(include_top=False, weights=None, input_shape=x.shape[1:])
            if self.weight_dir:
                tf.print("==> Loading pretrained weights", x.shape)
                net.load_weights(os.path.join(self.weight_dir, "flir_pretrained_weights.h5"), by_name=True)
        
        x = net(x)
        if fc:
            return self.fc(x)
        return x

    def fc(self, x):
        x = K.layers.Flatten()(x)
        x = K.layers.Dense(self.num_classes, activation="softmax")(x)
        return x
