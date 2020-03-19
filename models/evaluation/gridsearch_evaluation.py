import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = "/Users/lindronics/workspace/4th_year/out/out"

hists = {}
for path, _, files in os.walk(OUT_PATH):
    for fname in files:
        if fname.endswith(".pickle"):
            name = fname.split("_")
            model = name[0]
            config = name[1]

            with open(os.path.join(path, fname), "rb") as f:
                hist = pickle.load(f)

            hists[model] = hists.get(model, {})
            hists[model][config] = hist


for model in hists:
    fig, ((ax_loss, ax_acc), (ax_val_loss, ax_val_acc)) = plt.subplots(2, 2)

    for config, hist in hists[model].items():
        ax_loss.plot(hist["loss"], label=config)
        ax_acc.plot(hist["accuracy"], label=config)
        ax_val_loss.plot(hist["val_loss"], label=config)
        ax_val_acc.plot(hist["val_accuracy"], label=config)

    ax_loss.set_title("Training loss")
    ax_loss.legend()

    ax_acc.set_title("Training accuracy")
    ax_acc.legend()

    ax_val_loss.set_title("Validation loss")
    ax_val_loss.legend()

    ax_val_acc.set_title("Validation accuracy")
    ax_val_acc.legend()

    fig.suptitle(model)

    plt.show()
