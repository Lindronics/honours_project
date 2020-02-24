from os import listdir
from os.path import join, isdir


def generate_labels(name, filter_fn, root="images", label_path="", channel_prefix=True):
    """ Generates labels to be read by model data loader """

    classes = [c for c in listdir(root) if isdir(join(root, c))]
    samples = {c: 0 for c in classes}
    
    with open(label_path, "w") as f:
        for c in classes:
            batches = [b for b in listdir(join(root, c)) if isdir(join(root, c, b)) and filter_fn(b)]
            
            for b in batches:
                lwir_images = [i for i in listdir(join(root, c, b, "lwir")) if i.endswith(".png")]
                rgb_images = ["rgb_" + i[4:] for i in lwir_images] if channel_prefix else [i for i in lwir_images]

                lwir_paths = [join(root, c, b, "lwir", i) for i in lwir_images]
                rgb_paths = [join(root, c, b, "rgb", i) for i in rgb_images]

                for lwir_path, rgb_path in zip(lwir_paths, rgb_paths):
                    f.write(" ".join([rgb_path, lwir_path, c]) + "\n")
                    samples[c] += 1

    print(f"{name} dataset composition:")
    sum_ = 0
    for c, s in samples.items():
        sum_ += s
        print(f" - {c}: \t{s}")
    print(f"{sum_} items total.")
