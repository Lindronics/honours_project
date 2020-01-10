import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

from yolov3 import get_anchors, bbox_iou

class Dataset():

    def __init__(self, config, training=False):

        self.input_size = config["TRAINING"]["INPUT_SIZE"]
        self.channels = config["TRAINING"]["CHANNELS"]
        self.num_classes = config["NETWORK"]["NUM_CLASSES"]
        self.anchors_per_scale = config["NETWORK"]["ANCHORS_PER_SCALE"]
        self.strides = np.array(config["NETWORK"]["STRIDES"])
        self.output_sizes = self.input_size // self.strides
        self.max_bounding_box_per_scale = 150
        self.anchors = get_anchors("anchors.txt")

        self.load_metadata(config["TRAINING"]["ANNOTATIONS_DIR"])

        # For iterator
        self.current_batch = 0
        self.batch_size = config["TRAINING"]["BATCH_SIZE"] if training else config["TEST"]["BATCH_SIZE"]
        self.batches = len(self.annotations) // self.batch_size

    
    def load_metadata(self, path):
        """ Loads annotations and file paths """
        self.paths = []
        self.annotations = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                
                rgb_path = line[0]

                if len(line) > 1:
                    bounding_boxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
                else:
                    bounding_boxes = []
                
                self.annotations.append({
                    "rgb_path": rgb_path, 
                    "bounding_boxes": bounding_boxes,
                })

        print(self.paths)
        print(self.annotations)


    def __iter__(self):
        return self


    def __len__(self):
        return self.batches


    def __next__(self):
        """ Next data item """

        # Initialize batch tensors
        batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, self.channels), dtype=np.float32)

        batch_label_sbbox = np.zeros((self.batch_size, self.output_sizes[0], self.output_sizes[0],
                                      self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)
        batch_label_mbbox = np.zeros((self.batch_size, self.output_sizes[1], self.output_sizes[1],
                                      self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)
        batch_label_lbbox = np.zeros((self.batch_size, self.output_sizes[2], self.output_sizes[2],
                                      self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)

        batch_sbboxes = np.zeros((self.batch_size, self.max_bounding_box_per_scale, 4), dtype=np.float32)
        batch_mbboxes = np.zeros((self.batch_size, self.max_bounding_box_per_scale, 4), dtype=np.float32)
        batch_lbboxes = np.zeros((self.batch_size, self.max_bounding_box_per_scale, 4), dtype=np.float32)

        # Get items in batch
        if self.current_batch < self.batches:
            for i in range(self.batch_size):
                idx = self.current_batch * self.batch_size + i

                # Load and rescale image and bounding boxes
                rgb_path = self.annotations[idx]["rgb_path"]
                bounding_boxes = self.annotations[idx]["bounding_boxes"]

                image, scale = self.load_image(rgb_path)
                bounding_boxes = self.preprocess_bounding_boxes(bounding_boxes, scale)

                # Add to batch tensors
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bounding_boxes)
                batch_image[i, ...] = image
                batch_label_sbbox[i, ...] = label_sbbox
                batch_label_mbbox[i, ...] = label_mbbox
                batch_label_lbbox[i, ...] = label_lbbox
                batch_sbboxes[i, ...] = sbboxes
                batch_mbboxes[i, ...] = mbboxes
                batch_lbboxes[i, ...] = lbboxes

            self.current_batch += 1
            return batch_image, ((batch_label_sbbox, batch_sbboxes), (batch_label_mbbox, batch_mbboxes), (batch_label_lbbox, batch_lbboxes))


        # Shuffle dataset and reset iterator
        else:
            self.current_batch = 0
            np.random.shuffle(self.annotations)
            raise StopIteration


    def load_image(self, path):
        """ Loads and rescales image """

        image = cv2.imread(path) / 255
        h, w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        nh, nw, _ = image.shape
        return image, (nh/h, nw/w)


    def preprocess_bounding_boxes(self, bounding_boxes, scale):
        """ Rescales bounding boxes according to image scaling factor """

        h_scale, w_scale = scale
        
        if bounding_boxes == []:
            return None

        bounding_boxes[:, [0, 2]] = bounding_boxes[:, [0, 2]] * w_scale
        bounding_boxes[:, [1, 3]] = bounding_boxes[:, [1, 3]] * h_scale

        return bounding_boxes
        

    def preprocess_true_boxes(self, bounding_boxes):
        """ From https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3 """

        label = [np.zeros((self.output_sizes[i], self.output_sizes[i], self.anchors_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bounding_boxes_xywh = [np.zeros((self.max_bounding_box_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bounding_boxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bounding_box_per_scale)
                    bounding_boxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchors_per_scale)
                best_anchor = int(best_anchor_ind % self.anchors_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bounding_box_per_scale)
                bounding_boxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bounding_boxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

with open("config.json", "r") as f:
    config = json.load(f)

# d = Dataset(config, training=True)
# for image, annot in d:
    # print(image)
    # print(annot)
    # plt.imshow(image)
    # plt.show()