import json
import numpy as np
import tensorflow as tf
import os
import shutil

from yolov3 import *
from data import Dataset

logdir = "./data/log"

# Load configuration
with open("config.json", "r") as f:
    CONFIG = json.load(f)

TRAIN_CFG = CONFIG["TRAINING"]

# Load dataset
data = Dataset(CONFIG, training=True)

STEPS_PER_EPOCH = len(data)
WARMUP_STEPS = TRAIN_CFG["WARMUP_EPOCHS"] * STEPS_PER_EPOCH
TOTAL_STEPS = TRAIN_CFG["EPOCHS"] * STEPS_PER_EPOCH

# Prepare model
input_shape = [
    TRAIN_CFG["INPUT_SIZE"], 
    TRAIN_CFG["INPUT_SIZE"],
    TRAIN_CFG["CHANNELS"]
]
input_tensor = tf.keras.layers.Input(input_shape)
raw_prediction_tensors = yolo_v3(input_tensor)

prediction_tensors = []
for i, tensor in enumerate(raw_prediction_tensors):
    prediction_tensor = decode_predictions(tensor, i)
    prediction_tensors.append(tensor)
    prediction_tensors.append(prediction_tensor)

model = tf.keras.Model(input_tensor, prediction_tensors)
optimizer = tf.keras.optimizers.Adam()

if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

def train_step(X, y, step):

    with tf.GradientTape() as tape:

        # Get predictions
        predictions = model(X, training=True)

        # Calculate losses
        giou_loss = 0
        confidence_loss = 0
        probability_loss = 0

        for i in range(3):
            conv, pred = predictions[i*2], predictions[i*2+1]
            loss_items = compute_loss(pred, conv, *y[i], i)
            giou_loss += loss_items[0]
            confidence_loss += loss_items[1]
            probability_loss += loss_items[2]

        total_loss = giou_loss + confidence_loss + probability_loss

        # Optimize
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update learning rate
        if step < WARMUP_STEPS:
            learning_rate = step / WARMUP_STEPS * TRAIN_CFG["LEARNING_RATE_INIT"]
        else:
            learning_rate = TRAIN_CFG["LEARNING_RATE_END"] \
                + 0.5 * (TRAIN_CFG["LEARNING_RATE_INIT"] - TRAIN_CFG["LEARNING_RATE_END"]) \
                * (1 + tf.cos((step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS) * np.pi))
        optimizer.lr.assign(learning_rate)

        print(f"Step {step}, loss: {total_loss} ({giou_loss}, {confidence_loss}, {probability_loss}), learning rate: {learning_rate}")

        # Write to tensorboard
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=step)
            tf.summary.scalar("loss/total_loss", total_loss, step=step)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=step)
            tf.summary.scalar("loss/confidence_loss", confidence_loss, step=step)
            tf.summary.scalar("loss/probability_loss", probability_loss, step=step)
        writer.flush()

step = 0
for epoch in range(TRAIN_CFG["EPOCHS"]):
    for X, y in data:
        train_step(X, y, step)
        step += 1
    model.save_weights("./yolov3")
    print(f"Finished training epoch {epoch}.")
