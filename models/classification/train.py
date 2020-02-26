from os.path import join
import os
import shutil

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

from models.classification import alexnet
from models.classification.dataset import Dataset
from preprocessing.augment import augment_dataset
from preprocessing.generate_labels import generate_labels

# Change working directory
os.chdir('/nfs/honours_project')

train_labels_path = "../honours_project_data/animals_train_cleaned/labels.txt"
test_labels_path = "../honours_project_data/animals_test_cleaned/labels.txt"

print("Pre-processing dataset")
# filter_fn = lambda x: True
# generate_labels("train",  filter_fn, join(train_data_path, "images_160x120_augmented"), train_labels_path, channel_prefix=False)

train_data = Dataset(train_labels_path, rgb_only=False, res=(120, 160), register=False, batch_size=16, split=False)
test_data = Dataset(test_labels_path, rgb_only=False, res=(120, 160), register=False, batch_size=16, split=False)

X_train, y_train = train_data.get_all()
X_test, y_test = test_data.get_all()

print("Loading model")
model = alexnet.get_model(train_data)

print("Fitting model")
tf.random.set_seed(42)
np.random.seed(42)

model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

print("Finished training")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_ = test_data.get_labels()[:y_pred.shape[0]]
print(classification_report(y_test_, y_pred, target_names=test_data.class_labels))