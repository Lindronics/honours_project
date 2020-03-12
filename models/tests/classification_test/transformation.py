import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import cv2
import matplotlib.pyplot as plt

data_path="../../data/person"

rgb_shape = (1440, 1080)
fir_shape = (640, 480)

scaling_factor = rgb_shape[0] / fir_shape[0]

# Load calibration points
metadata_path = os.path.join(data_path, "metadata.json")
with open(metadata_path, "r") as f:
    metadata = json.load(f)

X = np.vstack(metadata["points"]["rgb"]) // scaling_factor
y = np.vstack(metadata["points"]["fir"])

print(X)
print(y)

# Fit model
model = LinearRegression()
model.fit(X, y)
transformation = np.hstack([model.coef_, model.intercept_[:, None]])

# Display fit
image_name = sorted(list(metadata["labels"].keys()))[0]
fir = cv2.imread(os.path.join(data_path, "fir", "fir_" + image_name)) / 255
rgb = cv2.imread(os.path.join(data_path, "rgb", "rgb_" + image_name)) / 255

fir = np.mean(fir, 2)
rgb = np.mean(rgb, 2)

rgb = cv2.resize(rgb, (fir.shape[1], fir.shape[0]))
rgb = cv2.warpAffine(rgb, transformation, (fir.shape[1], fir.shape[0]))

superimposed = np.mean(np.dstack([rgb, fir]), axis=2)
plt.imshow(superimposed)
plt.show()

# Save transformation matrix
with open("transformation.txt", "w") as f:
    f.write(f"{fir_shape[0]},{fir_shape[1]}\n")
    for row in transformation:
        s = ",".join(np.char.mod("%f", row)) + "\n"
        f.write(s)