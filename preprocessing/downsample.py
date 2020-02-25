from os import listdir, makedirs
from os.path import isdir, join, exists
from shutil import rmtree
import cv2
from tqdm import tqdm


def change_path(path, new_dir):
    path = path.split("/")[1:]
    return join(new_dir, *path)


def downsample(root:str, new_dir:str, res:tuple=(480, 640)):
    """ 
    Downsamples dataset to a given resolution and stores it at new_dir 
    
    Params
    ------
    res: tuple
        Resolution to resize to (width, height)
    root: str
        Path of the dataset folder containing the class directories
    new_dir: str
        Directory to store downsampled data at
    """

    if exists(new_dir):
        rmtree(new_dir)

    classes = [c for c in listdir(root) if isdir(join(root, c))]
    
    for c in tqdm(classes):
        batches = [b for b in listdir(join(root, c)) if isdir(join(root, c, b))]
        
        for b in batches:
            makedirs(join(new_dir, c, b, "lwir"))
            makedirs(join(new_dir, c, b, "rgb"))

            lwir_images = [i for i in listdir(join(root, c, b, "lwir")) if i.endswith(".png")]
            rgb_images = ["rgb_" + i[4:] for i in lwir_images]

            lwir_paths = [join(root, c, b, "lwir", i) for i in lwir_images]
            rgb_paths = [join(root, c, b, "rgb", i) for i in rgb_images]

            for lwir_path, rgb_path in zip(lwir_paths, rgb_paths):
                lwir = cv2.imread(lwir_path)
                lwir = cv2.resize(lwir, res)
                cv2.imwrite(change_path(lwir_path, new_dir), lwir)

                rgb = cv2.imread(rgb_path)
                rgb = cv2.resize(rgb, res)
                cv2.imwrite(change_path(rgb_path, new_dir), rgb)

    print(f"Successfully downsampled dataset to {res}")
