from os.path import join
import shutil

import tensorflow as tf
import numpy as np

from models.classification import alexnet
from models.classification.dataset import Dataset
from preprocessing.augment import augment_dataset
from preprocessing.generate_labels import generate_labels

data_dir = "../honours_project_data/animals"
labels_path = join(data_dir, "labels.txt")

print("Pre-processing dataset")
class_whitelist = [
    "pony",
    "alpaca",
    "chicken",
    "peacock",
    "cat",
]
filter_fn = lambda x: x.split("_")[0] in class_whitelist
generate_labels("train",  filter_fn, join(data_dir, "images_160x120_downsampled"), labels_path)

train_data = Dataset(labels_path, rgb_only=False, res=(120, 160), register=True, batch_size=16, split=False)

print("Loading model")
model = alexnet.get_model(train_data)

print("Fitting model")
tf.random.set_seed(42)
np.random.seed(42)

model.fit(train_data, epochs=30)

print("Finished training")