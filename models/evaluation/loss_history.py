import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

HISTS_PATH = "/Users/lindronics/workspace/4th_year/out/stratified/stratified_hists.pickle"

with open(HISTS_PATH, "rb") as f:
    hists = pickle.load(f)

font = {
    "fontname": "fbb",
    # "fontweight": "bold",
}

# fig, (ax_loss, ax_acc) = plt.subplots(1, 1, dpi=150)
fig, ax_acc = plt.subplots(1, 1, dpi=150, figsize=(3.3, 2.2))

# ax_loss.plot(hists["loss"], label="Training")
ax_acc.plot(hists["accuracy"], label="Training")
# ax_loss.plot(hists["val_loss"], label="Validation")
ax_acc.plot(hists["val_accuracy"], label="Validation")

# ax_loss.set_title("Loss", fontdict=font)

# ax_acc.set_title("Accuracy", fontdict=font)
ax_acc.legend()

plt.savefig("stratified.pdf")
plt.show()
