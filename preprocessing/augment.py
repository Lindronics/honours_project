import os
import shutil
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tqdm
import argparse

from preprocessing.generate_labels import generate_labels


transformation = np.array([
    [1.202290, -0.026808, -50.528589],
    [0.017762, 1.203090, -73.950204],
])

def save(path, name, rgb, lwir, label, extension=""):
    rgb_path = os.path.join(path, "rgb", name + f"_{extension}.png")
    lwir_path = os.path.join(path, "lwir", name + f"_{extension}.png")

    cv2.imwrite(rgb_path, rgb)
    cv2.imwrite(lwir_path, lwir)
    return " ".join([rgb_path, lwir_path, label])


def augment_dataset(metadata, augmented_dir, multiplier=2, register=False):

    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=30,
        zoom_range=[0.5, 1],
    )

    classes = dict()
    samples = list()

    # Load labels
    with open(metadata, "r") as f:
        for line in f:
            line = line.split()
            samples.append(line)

            classes[line[-1]] = classes.get(line[-1], 0) + 1

    # Get maximum class samples
    max_class = max(classes.items(), key=lambda x: x[1])
    class_multipliers = {x: max_class[1] * multiplier // y for x, y in classes.items()}
    class_remainders = {x: (max_class[1] * multiplier) % y for x, y in classes.items()}

    # Generate output directory structure
    for label in classes:
        os.makedirs(os.path.join(augmented_dir, label, f"{label}_single_1", "lwir"))
        os.makedirs(os.path.join(augmented_dir, label, f"{label}_single_1", "rgb"))

    # Augment and copy
    augmented_samples = []
    for rgb_path, lwir_path, label in tqdm(samples, position=0):
        rgb = cv2.imread(rgb_path)

        if register:
            initial_shape = rgb.shape
            rgb = cv2.resize(rgb, (480, 640))
            rgb = cv2.warpAffine(rgb, transformation, (480, 640))
            rgb = cv2.resize(rgb, (initial_shape[1], initial_shape[0]))

        lwir = cv2.imread(lwir_path)

        name = (rgb_path.split("/")[-1]).split(".")[0][4:]
        out_path = os.path.join(augmented_dir, label, f"{label}_single_1")

        # Copy original
        augmented_samples.append(save(out_path, name, rgb, lwir, label, extension=""))

        # Flipped
        rgb_t = datagen.apply_transform(rgb, {"flip_horizontal": True})
        lwir_t = datagen.apply_transform(lwir, {"flip_horizontal": True})

        augmented_samples.append(save(out_path, name, rgb_t, lwir_t, label, extension=0))

        # Random transformations
        remainder = 1 if class_remainders[label] > 0 else 0
        class_remainders[label] -= 1
        for i in range(2, class_multipliers[label] + remainder):
            trans = datagen.get_random_transform(rgb.shape)
            rgb_t = datagen.apply_transform(rgb, trans)
            lwir_t = datagen.apply_transform(lwir, trans)

            augmented_samples.append(save(out_path, name, rgb_t, lwir_t, label, extension=i))


    with open("augmented.txt", "w") as f:
        for line in augmented_samples:
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment and pre-process dataset")
    parser.add_argument("in_path", help="Input directory containing images")
    parser.add_argument("out_path", help="Output directory for augmented images")
    parser.add_argument("labels_path", help="Output file path for labels file")
    args = vars(parser.parse_args())

    generate_labels("Full", lambda x: True, args["in_path"], args["labels_path"], channel_prefix=True)

    augment_dataset(args["labels_path"], args["out_path"], multiplier=4, register=True)
    generate_labels("Full augmented", lambda x: True, args["out_path"], args["labels_path"], channel_prefix=False)
    
    print("\nFinished.")
