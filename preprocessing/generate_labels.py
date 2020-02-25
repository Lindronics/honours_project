from os import listdir
from os.path import join, isdir


def generate_labels(filter_fn:function, root:str, channel_prefix:bool=True):
    """ 
    Generates labels to be read by Dataset loader
    
    Params
    ------
    filter_fn: function       
        Function accepting the name of the data subset/batch 
        and returning True if the batch shall be processed
    root: str
        Path of the dataset folder containing the class directories
    channel_prefix: boolean
        Whether the images in the dataset are prefixed with RGB_ and FIR_.

    Returns
    -------
    A list containing tuples of (rgb path, lwir path, class name)
    """

    classes = [c for c in listdir(root) if isdir(join(root, c))]
    samples = {c: 0 for c in classes}
    
    samples = []
    for c in classes:
        batches = [b for b in listdir(join(root, c)) if isdir(join(root, c, b)) and filter_fn(b)]
        
        for b in batches:
            lwir_images = [i for i in listdir(join(root, c, b, "lwir")) if i.endswith(".png")]
            rgb_images = ["rgb_" + i[4:] for i in lwir_images] if channel_prefix else [i for i in lwir_images]

            lwir_paths = [join(root, c, b, "lwir", i) for i in lwir_images]
            rgb_paths = [join(root, c, b, "rgb", i) for i in rgb_images]

            for lwir_path, rgb_path in zip(lwir_paths, rgb_paths):
                samples.append(tuple([rgb_path, lwir_path, c]))
                samples[c] += 1

    print(f"Dataset composition:")
    sum_ = 0
    for c, s in samples.items():
        sum_ += s
        print(f" - {c}: \t{s}")
    print(f"{sum_} items total.")

    return samples


def write_labels(labels: list, path: str):
    """
    Writes labels to a file.
    
    Params
    ------
    labels: list
        List containing label tuples
    path: str
        Path to write to
    """
    with open(path, "w") as f:
        for label in labels:
            f.write(" ".join(label) + "\n")
