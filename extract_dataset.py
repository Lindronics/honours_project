import os
import shutil
import argparse
import numpy as np
import flirimageextractor
from skimage.io import imsave


def extract_image(flir, path, verbose=False):
    if verbose:
        print(f"Extracting image {path}.")
    flir.process_image(path)
    thermal = flir.get_thermal_np()
    rgb = flir.extract_embedded_image()
    return thermal, rgb


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract thermal and visible data from FLIR One exported images.")
    parser.add_argument("in_path", help="Input directory containing exported images")
    parser.add_argument("out_path", help="Output directory for extracted dataset")
    parser.add_argument("-v", "--verbose", dest="verbose", help="Verbose output", action="store_true")
    args = vars(parser.parse_args())

    # Setup output dir  
    shutil.rmtree(args["out_path"], ignore_errors=False, onerror=None)
    os.makedirs(os.path.join(args["out_path"], "thermal"))
    os.makedirs(os.path.join(args["out_path"], "visible"))

    flir = flirimageextractor.FlirImageExtractor(palettes=[])

    images = filter(lambda f: f.endswith(".jpg"), os.listdir(args["in_path"]))
    for image in images:
        thermal, visible = extract_image(flir, os.path.join(args["in_path"], image), args["verbose"])
        imsave(os.path.join(args["out_path"], "thermal", image), thermal)
        imsave(os.path.join(args["out_path"], "visible", image), visible)
