import tensorflow as tf
import tensorflow.keras as K

def net(x, train_data):

    ##### Split VIS and LWIR data
    
    rgb = x[..., 0:3]
    lwir = x[..., 3, None]

    ##### VIS Path

    rgb = K.layers.Conv2D(filters=96, kernel_size=11, strides=2, padding="same", activation="relu")(rgb)
    rgb = K.layers.MaxPool2D(pool_size=3, strides=2)(rgb)

    rgb = K.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="same", activation="relu")(rgb)
    rgb = K.layers.MaxPool2D(pool_size=3, strides=2)(rgb)

    rgb = K.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same", activation="relu")(rgb)
    rgb = K.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same", activation="relu")(rgb)
    rgb = K.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(rgb)
    rgb = K.layers.MaxPool2D(pool_size=3, strides=2)(rgb)

    rgb = K.layers.Flatten()(rgb)
    rgb = K.layers.Dense(1024, activation="relu")(rgb)
    rgb = K.layers.Dense(1024, activation="relu")(rgb)
    
    if train_data.rgb_only:
        rgb = K.layers.Dense(train_data.num_classes(), activation="softmax")(rgb)
        return rgb

    ##### LWIR Path

    lwir = K.layers.Conv2D(filters=96, kernel_size=11, strides=2, padding="same", activation="relu")(lwir)
    lwir = K.layers.MaxPool2D(pool_size=3, strides=2)(lwir)

    lwir = K.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="same", activation="relu")(lwir)
    lwir = K.layers.MaxPool2D(pool_size=3, strides=2)(lwir)

    lwir = K.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same", activation="relu")(lwir)
    lwir = K.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same", activation="relu")(lwir)
    lwir = K.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(lwir)
    lwir = K.layers.MaxPool2D(pool_size=3, strides=2)(lwir)

    lwir = K.layers.Flatten()(lwir)
    lwir = K.layers.Dense(1024, activation="relu")(lwir)
    lwir = K.layers.Dense(1024, activation="relu")(lwir)

    ##### Joint output

    x = K.layers.Concatenate()([rgb, lwir])
    tf.print(x.shape)
    x = K.layers.Dense(train_data.num_classes(), activation="softmax")(x)
    return x


def get_model(train_data):
    input_tensor = K.layers.Input(train_data.shape())
    output_tensor = net(input_tensor, train_data)

    model = K.Model(input_tensor, output_tensor)
    model.compile(optimizer='sgd',
                loss='mean_squared_error',
                metrics=['accuracy'])
    model.summary()
    return model