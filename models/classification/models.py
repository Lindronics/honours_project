import tensorflow as tf
import tensorflow.keras as K


class AbstractModel():
    """ 
    Abstract model class. Provides an interface for all models.
    """

    modes = ["rgb", "lwir", "grayscale", "stacked", "voting", "fusion"]

    def __init__(self, mode, num_classes):
        """
        Params
        ------
        mode: str
            Multispectral mode of the model. 
            Can be rgb, lwir, grayscale, stacked, voting or fusion.
        num_classes: int
            Number of classes
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
        
    def __call__(self, x):
        return self.method(x)

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
        x = x[..., 3, None]
        x = self.net(x)
        return x

    def lwir(self, x):
        x = x[..., 0:3]
        x = self.net(x)
        return x

    def grayscale(self, x):
        x = x[..., 0:3]
        x = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        x = self.net(x)
        return x

    def stacked(self, x):
        x = self.net(x)
        return x

    def voting(self, x):
        rgb = self.net(x)
        lwir = self.net(x)
        x = K.layers.Add()([rgb, lwir]) * 0.5
        return x

    def fusion(self, x):
        rgb = self.net(x, fc=False)
        lwir = self.net(x, fc=False)
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
