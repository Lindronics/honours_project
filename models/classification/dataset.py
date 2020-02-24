import numpy as np
# import cv2
from itertools import count
from collections import defaultdict
from sklearn.model_selection import train_test_split
import tensorflow as tf


transformation = np.array([
    [1.202290, -0.026808, -50.528589],
    [0.017762, 1.203090, -73.950204],
])

class Dataset(K.utils.Sequence):

    def __init__(self, metadata, rgb_only=False, res=(480, 640), register=True, batch_size=16, split=False, train=True):

        self.rgb_only = rgb_only
        self.res = res
        self.register = register
        self.batch_size = batch_size

        self.classes = defaultdict(count().__next__)
        self.samples = []

        samples = []
        labels = []
        with open(metadata, "r") as f:
            for line in f:
                line = line.split()
                class_label = self.classes[line[-1]]
                samples.append(tuple(line[:2] + [class_label]))
                labels.append(class_label)

        if split:
            train_data, test_data = train_test_split(samples, stratify=labels, test_size=0.2, random_state=42)
            self.samples = train_data if train else test_data
        else:
            self.samples = samples

        self.class_dict = {key: val for val, key in self.classes.items()}
        self.class_labels = [val for key, val in sorted(self.class_dict.items())]

    def __len__(self):
        return len(self.samples) // self.batch_size
    

    def __iter__(self):
        return self


    def load(self, path, register=False):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        # img = cv2.imread(path) / 255
        # if register:
        #     img = cv2.resize(img, (480, 640))
        #     img = cv2.warpAffine(img, transformation, (480, 640))
        # img = cv2.resize(img, self.res)
        return img


    def __getitem__(self, idx):
        if idx < self.__len__():
            X = []
            y = []
            for i in range(self.batch_size):
                rgb_path, lwir_path, label = self.samples[idx * self.batch_size + i]

                one_hot = np.zeros(self.num_classes())
                one_hot[label] = 1
                y.append(one_hot)

                rgb = self.load(rgb_path, register=self.register)

                if self.rgb_only:
                    X.append(rgb)
                else:
                    lwir = self.load(lwir_path, register=False)
                    lwir = np.mean(lwir, -1)[..., None]
                    stacked = np.dstack([rgb, lwir])
                    X.append(stacked)

            return np.array(X), np.array(y), [None]
        else:
            raise StopIteration

    def get_labels(self):
        return np.array([sample[-1] for sample in self.samples])

    def num_classes(self):
        return len(self.classes)

    def shape(self):
        return self.res[::-1] + tuple([(3 if self.rgb_only else 4)])
