import os
import pickle
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = "/Users/lindronics/workspace/4th_year/out/kfold"

final_report = {}
for path, _, files in os.walk(OUT_PATH):
    for fname in files:
        if fname.endswith(".pickle") and "report" in fname:

            with open(os.path.join(path, fname), "rb") as f:
                report = pickle.load(f)

            for c, metrics in report.items():
                if c == "accuracy":
                    continue
                final_report[c] = final_report.get(c, {})
                for metric, val in metrics.items():
                    final_report[c][metric] = final_report[c].get(metric, [])
                    final_report[c][metric].append(val)

for c, metrics in final_report.items():
    for metric, vals in metrics.items():
        final_report[c][metric] = f"{np.mean(np.array(vals)):.2f}"


pprint(final_report)


