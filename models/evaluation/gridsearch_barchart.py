import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = "/Users/lindronics/workspace/4th_year/out/new_out"

scores = {}
for model in os.listdir(OUT_PATH):
    if not os.path.isdir(os.path.join(OUT_PATH, model)):
        continue
    scores[model] = []
    report_path = os.path.join(OUT_PATH, model, "report.txt")
    with open(report_path, "r") as f:
        for line in f:
            if "weighted avg" in line:
                scores[model].append(float(line.split()[-2]))

x = np.arange(0, 6)
configs = ["rgb", "lwir", "gray", "stacked", "voting", "fusion"]

print(scores)
fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
ax.bar(x * (len(scores) + 1) + 0, scores["CustomNet"], label="CustomNet")
ax.bar(x * (len(scores) + 1) + 1, scores["AlexNet"], label="AlexNet", tick_label=configs)
ax.bar(x * (len(scores) + 1) + 2, scores["ResNet"], label="ResNet")
ax.legend(loc="lower right", framealpha=0.9)
plt.savefig("output.pdf")
plt.show()
