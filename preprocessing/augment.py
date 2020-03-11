import os
import shutil
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import argparse

from generate_labels import generate_labels, write_labels
from trans_cfg import cfg


def save(path:str, name:str, rgb:np.ndarray, lwir:np.ndarray, label:str, extension:str="") -> tuple:
    """
    Saves image to a path.

    Params
    ------
    path: str
        Subset/batch directory to save to
    name: str
        Name of the image
    rgb: np.ndarray
        RGB image
    lwir: np.ndarray
        LWIR image
    label: str
        Class label
    extension: str
        String to be added to end of filename (e.g. number)

    Returns
    -------
    Tuple of new (rgb path, lwir path, class label)
    """
    rgb_path = os.path.join(path, "rgb", name + f"_{extension}.png")
    lwir_path = os.path.join(path, "lwir", name + f"_{extension}.png")

    cv2.imwrite(rgb_path, rgb)
    cv2.imwrite(lwir_path, lwir)
    return tuple([rgb_path, lwir_path, label])


def augment_dataset(samples: list, augmented_dir:str, multiplier:int=2, register:bool=False, res:tuple=(120, 160)):
    """ 
    Augments a dataset. Will attempt to balance classes.

    Params
    ------
    samples: list
        List of sample annotation tuples (rgb path, lwir path, class label)
    augmented_dir: str
        Directory to store augmented data at
    multiplier: int
        Multiplication factor for number of new samples
    register: bool
        Whether to align RGB images with LWIR images

    Returns
    ------
    A list containing tuples of (rgb path, lwir path, class name) for new augmented dataset
    """

    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=30,
        zoom_range=[0.5, 1],
    )

    classes = {}

    # Get classes
    for sample in samples:
        classes[sample[-1]] = classes.get(sample[-1], 0) + 1

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
            rgb = cv2.resize(rgb, cfg.small.res)
            rgb = cv2.warpAffine(rgb, cfg.small.matrix, cfg.small.res)
        rgb = cv2.resize(rgb, res)

        lwir = cv2.imread(lwir_path)
        lwir = cv2.resize(lwir, res)

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

    return augmented_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment and pre-process dataset")
    parser.add_argument("in_path", help="Input directory containing images")
    parser.add_argument("out_path", help="Output directory for augmented images")
    parser.add_argument("labels_path", help="Output file path for labels file")
    parser.add_argument("multiplier", help="Multiplier")
    parser.add_argument("-d", "--downsample", dest="downsample", help="Downsample to 160x120", action="store_true")
    args = vars(parser.parse_args())

    res = (120, 160) if args["downsample"] else (480, 640)

    labels = generate_labels(lambda x: True, args["in_path"], channel_prefix=True)
    new_labels = augment_dataset(labels, args["out_path"], multiplier=int(args["multiplier"]), register=True, res=res)
    write_labels(new_labels, args["labels_path"])
    
    print("\nFinished.")
