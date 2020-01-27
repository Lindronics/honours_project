import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, MaxPool2D, Dropout, Flatten
from tensorflow.keras.losses import categorical_crossentropy


def get_model(input_shape, num_categories):
    model = Sequential()
    
    model.add(InputLayer(input_shape))
    
    # Convolution 1
    model.add(Conv2D(filters=96, kernel_size=11, strides=4, padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=3, strides=2, padding="valid"))

    # Convolution 2
    model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=3, strides=2))

    # Convolution 3
    model.add(Conv2D(filters=384, kernel_size=3, strides=1, padding="same", activation="relu"))

    # Convolution 4
    model.add(Conv2D(filters=384, kernel_size=3, strides=1, padding="same", activation="relu"))

    # Convolution 5
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=3, strides=2))

    # Flatten for fully connected
    model.add(Flatten())

    # Fully connected 6
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.4))

    # Fully connected 7
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.4))

    # Fully connected 8
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.4))

    # Output layer
    model.add(Dense(num_categories, activation="softmax"))
    return model


def get_simple_model(input_shape, num_categories):
    model = Sequential()
    
    model.add(InputLayer(input_shape))

    # Convolution 1
    model.add(Conv2D(filters=32, kernel_size=11, strides=2, padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=11, strides=3))

    # Convolution 2
    model.add(Conv2D(filters=32, kernel_size=7, strides=2, padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=5, strides=2))

    # Convolution 3
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=4, strides=2))

    # Flatten for fully connected
    model.add(Flatten())

    # Fully connected 4
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_categories, activation="sigmoid"))
    return model

m = get_simple_model((640, 480, 4), 1)
m.summary()


