import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Dataset():

    def __init__(self, config):
        self.config = config

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def __next__(self):
        image_path = "data/I01300.jpg"

        annots = np.array([[414,215,22,52,0], [332,215,20,49,0]])

        image = cv2.imread(image_path) / 255
        h, w, _ = image.shape
        image = cv2.resize(image, (416, 416))
        image = np.stack([image]*4, axis=0)

        # TODO adjust annotations to new scale

        asdfl = np.zeros(shape=(4, 52, 52, 3, 85), dtype=np.float32)
        asdfm = np.zeros(shape=(4, 26, 26, 3, 85), dtype=np.float32)
        asdfs = np.zeros(shape=(4, 13, 13, 3, 85), dtype=np.float32)
        asdf2 = np.zeros(shape=(4, 150, 4), dtype=np.float32)

        return image, [[asdfl, asdf2], [asdfm, asdf2], [asdfs, asdf2]]
        

with open("config.json", "r") as f:
    config = json.load(f)

# d = Dataset(config)
# for image, annot in d:
#     plt.imshow(image)
#     plt.show()