import os
import json
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from scipy.ndimage import affine_transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy

from model import get_model, get_simple_model


def get_data(data_path="../../data/person"):

    metadata_path = os.path.join(data_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    fir_dir = os.path.join(data_path, "fir")
    rgb_dir = os.path.join(data_path, "rgb")

    paths = sorted(os.listdir(fir_dir))
    paths = [p[4:] for p in paths]

    images = []
    labels = []
    for path in paths:
        fir = cv2.imread(os.path.join(fir_dir, "fir_" + path)) / 255
        rgb = cv2.imread(os.path.join(rgb_dir, "rgb_" + path)) / 255

        fir = np.mean(fir, 2)
        fir = cv2.resize(fir, (240, 320))
        rgb = cv2.resize(rgb, (fir.shape[1], fir.shape[0]))

        image = np.dstack([rgb, fir])
        images.append(image)
        labels.append(metadata["labels"][path])
    
    return np.array(images), np.array(labels)



if __name__ == "__main__":

    classes = ["nothing", "human"]
    
    images, labels = get_data()
    # enc = OneHotEncoder(sparse=False)
    # labels = enc.fit_transform(labels[:,None])

    print(images.shape)
    print(labels)
        
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

    model = get_simple_model(input_shape=images.shape[1:], num_categories=1)
    model.summary()
    # model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
    model.compile(loss=binary_crossentropy, optimizer="adam", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=23, validation_split=0.2)

    # y_pred = model.predict_classes(X_test)[:,None]
    y_pred = model.predict_classes(X_test)
    # print(classification_report(y_test, enc.transform(y_pred)))
    print(classification_report(y_test, y_pred, target_names=classes))

    model.save("test.h5")