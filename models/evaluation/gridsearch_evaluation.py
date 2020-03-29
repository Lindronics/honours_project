import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = "/Users/lindronics/workspace/4th_year/out/new_out"

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

font = {
    "fontname": "fbb",
    # "fontweight": "bold",
}
for model in hists:
    fig, ((ax_loss, ax_acc), (ax_val_loss, ax_val_acc)) = plt.subplots(2, 2, dpi=150)

    for config, hist in hists[model].items():
        if config == "grayscale":
            continue

        ax_loss.plot(hist["loss"], label=config)
        ax_acc.plot(hist["accuracy"], label=config)
        ax_val_loss.plot(hist["val_loss"], label=config)
        ax_val_acc.plot(hist["val_accuracy"], label=config)

    ax_loss.set_title("Training loss", fontdict=font)
    # ax_loss.legend()

    ax_acc.set_title("Training accuracy", fontdict=font)
    # ax_acc.legend()

    ax_val_loss.set_title("Validation loss", fontdict=font)
    # ax_val_loss.legend()

    ax_val_acc.set_title("Validation accuracy", fontdict=font)
    ax_val_acc.legend(prop={"size": 8})

    # fig.suptitle(model)
    fig.tight_layout()

    plt.savefig(f"{model}.pdf")
    plt.show()
