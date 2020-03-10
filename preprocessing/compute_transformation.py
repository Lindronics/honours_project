import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

anchors_path = "preprocessing/registration_anchors.json"
vis_res = np.array([1080, 1440])
lwir_res = np.array([460, 640])
target_res = np.array([120, 160])

def scale(points, res):
    return (points * target_res / res).astype(np.int32)

# Load points
with open(anchors_path, "r") as f:
    data = json.load(f)["images"]

images = [d["image"] for d in data]
subsets = [d["subset"] for d in data]
vis_points = [scale(np.array(d["vis"]), vis_res) for d in data]
lwir_points = [scale(np.array(d["lwir"]), lwir_res) for d in data]

# Concatenate
X = np.concatenate(vis_points, axis=0)
y = np.concatenate(lwir_points, axis=0)

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.scatter(X[:, 0], y[:, 0], alpha=0.3, label="horizontal")
ax.scatter(X[:, 1], y[:, 1], alpha=0.3, label="vertical")
ax.set_xlabel("Visible light coordinate")
ax.set_ylabel("LWIR coordinate")
ax.legend()
plt.show()

# Fit model
model = LinearRegression()
model.fit(X, y)

# Evaluate
loss = []
for subset, vis, lwir in zip(subsets, vis_points, lwir_points):
    y_pred = model.predict(vis)
    norms = np.linalg.norm(y_pred - lwir, axis=1)
    mean_distance = np.mean(norms)
    loss.append(mean_distance)

# plt.figure()
# plt.barh(np.arange(len(loss)), loss, tick_label=subsets)
# plt.show()

# Group by class
subset_classes = map(lambda x: x.split("_")[0], subsets)
df = pd.DataFrame(zip(subset_classes, loss), columns=["class", "loss"])
df = df.groupby("class").mean()

fig, ax = plt.subplots(figsize=(3.5, 2.5))
# plt.barh(np.arange(len(loss)), loss, tick_label=subsets)
ax.barh(np.arange(len(df)), df.loss, tick_label=df.index)
ax.set_xlabel("Mean euclidean distance")
plt.show()

print("Images:", len(images))
print("Points:", X.shape[0])
print("Mean loss:", np.mean(np.array(loss)))
print("b:", model.intercept_)
print("A:", model.coef_)