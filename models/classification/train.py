from os.path import join
import os
import shutil

import tensorflow as tf
import numpy as np

from models.classification import alexnet
from models.classification.dataset import Dataset
from preprocessing.augment import augment_dataset
from preprocessing.generate_labels import generate_labels

# Change working directory
os.chdir('/nfs/honours_project')

data_dir = "../honours_project_data/animals"
labels_path = join(data_dir, "augmented.txt")

print("Pre-processing dataset")
filter_fn = lambda x: True
generate_labels("train",  filter_fn, join(data_dir, "images_160x120_augmented"), labels_path, channel_prefix=False)

train_data = Dataset(labels_path, rgb_only=False, res=(120, 160), register=False, batch_size=16, split=False)

print("Loading model")
model = alexnet.get_model(train_data)

print("Fitting model")
tf.random.set_seed(42)
np.random.seed(42)

model.fit(train_data, epochs=30)

print("Finished training")