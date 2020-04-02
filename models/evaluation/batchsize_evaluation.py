import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = "/Users/lindronics/workspace/4th_year/out/batch_size"

hists = {}
for path, _, files in os.walk(OUT_PATH):
    for fname in files:
        if fname.endswith(".pickle"):
            batch_size = fname.split("_")[0]

            with open(os.path.join(path, fname), "rb") as f:
                hist = pickle.load(f)

            hists[int(batch_size)] = hist

font = {
    "fontname": "fbb",
    # "fontweight": "bold",
}
axis_font = {"size": "8"}

fig, ((ax_loss, ax_acc), (ax_val_loss, ax_val_acc)) = plt.subplots(2, 2, dpi=150)

for config, hist in sorted(hists.items()):
    ax_loss.plot(hist["loss"], label=config)
    ax_loss.set_xlabel("Training epoch", fontdict=axis_font)
    ax_acc.plot(hist["accuracy"], label=config)
    ax_acc.set_xlabel("Training epoch", fontdict=axis_font)

    ax_val_loss.plot(hist["val_loss"], label=config)
    ax_val_loss.set_xlabel("Training epoch", fontdict=axis_font)
    ax_val_acc.plot(hist["val_accuracy"], label=config)
    ax_val_acc.set_xlabel("Training epoch", fontdict=axis_font)

ax_loss.set_title("Training loss", fontdict=font)
# ax_loss.x_
# ax_loss.legend()

ax_acc.set_title("Training accuracy", fontdict=font)
# ax_acc.legend()

ax_val_loss.set_title("Validation loss", fontdict=font)
# ax_val_loss.legend()

ax_val_acc.set_title("Validation accuracy", fontdict=font)
ax_val_acc.legend()

# fig.suptitle("asdf")
fig.tight_layout()

fig.savefig("output.pdf")
plt.show()
