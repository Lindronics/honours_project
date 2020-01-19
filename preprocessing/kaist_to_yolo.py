import os

root_path = "data/kaist/"

in_path = os.path.join(root_path, "in")
out_path = os.path.join(root_path, "out")

# os.makedirs(out_path)

classes = {}
class_counter = 0

def get_class_id(class_name):
    global class_counter
    if class_name in classes:
        return classes[class_name]

    classes[class_name] = class_counter
    class_counter += 1
    return classes[class_name]


def parse_set_annotations(set_):
    new_annots = []

    annot_path = os.path.join(in_path, "annotations", set_)
    for subset in os.listdir(annot_path):
        if os.path.isfile(subset):
            continue

        for item in os.listdir(os.path.join(annot_path, subset)):
            with open(os.path.join(annot_path, subset, item), "r") as f:
                annots = f.readlines()[1:]

            if len(annots) == 0:
                continue

            rgb_path = os.path.join(in_path, "images", set_, subset, item[:-3] + "jpg")
            new_annot = [rgb_path]

            for line in annots:
                line = line.strip().split()
                class_id = get_class_id(line[0])
                l = line[1:5] + [str(class_id)]
                new_annot.append(",".join(l))

            new_annots.append(" ".join(new_annot))
    return new_annots

# Get all downloaded datasets
all_annots = []
for set_ in os.listdir(os.path.join(in_path, "images")):
    if not os.path.isfile(set_):
        all_annots += parse_set_annotations(set_)

# Save annotations
with open(os.path.join(out_path, "annotations.txt"), "w") as f:
    for line in all_annots:
        f.write(line + "\n")

# Save class names
classes = sorted(classes.items(), key=lambda x:x[1])
with open(os.path.join(out_path, "classes.txt"), "w") as f:
    for class_name, _ in classes:
        f.write(class_name + "\n")
