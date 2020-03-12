import os
import numpy as np
import cv2
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("test.h5")

def get_data(data_path="../../data/person"):

    fir_dir = os.path.join(data_path, "fir")
    rgb_dir = os.path.join(data_path, "rgb")

    paths = sorted(os.listdir(fir_dir))
    paths = [p[4:] for p in paths]

    images = []
    for path in paths:
        fir = cv2.imread(os.path.join(fir_dir, "fir_" + path)) / 255
        rgb = cv2.imread(os.path.join(rgb_dir, "rgb_" + path)) / 255

        fir = np.mean(fir, 2)
        fir = cv2.resize(fir, (240, 320))
        rgb = cv2.resize(rgb, (fir.shape[1], fir.shape[0]))

        image = np.dstack([rgb, fir])
        # plt.imshow(image)
        # plt.show()
        images.append(image)
    
    return np.array(images)

x = get_data(data_path="../../data/person_test")
# print(x.shape)
# plt.imshow(x[0, ...])
# plt.show()


y_pred = model.predict(x)
print(y_pred)
# print(classification_report(y, OneHotEncoder().fit_transform(y_pred[:, None])))
