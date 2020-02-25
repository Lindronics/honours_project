"""
Functionality for creating a YOLO v3 network.
Includes loading and creating the modules.
Based on https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.functional as F


class EmptyLayer(nn.Module):
    """ Empty dummy layer """

    def __init__(self) -> None:
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    """ Holds YOLO anchors for bounding boxes """

    def __init__(self, anchors) -> None:
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class NetCreator:
    """ Creates the modules for the YOLO v3 architecture. """

    def __init__(self, fname: str) -> None:
        self.blocks = self.read_cfg(fname)
        self.create_layer = {
            "convolutional": self.create_convolutional,
            "upsample": self.create_upsample,
            "route": self.create_route,
            "shortcut": self.create_shortcut,
            "yolo": self.create_yolo,
        }

    def read_cfg(self, fname: str) -> Dict:
        """ Reads and parses yolo v3 config file """

        with open(fname, "r") as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines if x.strip() and x[0] != "#"]

        
        current_block = {}
        blocks = []
        
        for line in lines:
            if line[0] == "[":       
                if len(current_block) != 0:
                    blocks.append(current_block)
                    current_block = {}
                current_block["type"] = line[1:-1].strip()     
            else:
                key, value = line.split("=") 
                current_block[key.strip()] = value.strip()

        blocks.append(current_block)
        
        return blocks

    def create_convolutional(self, index: int, config: Dict) -> nn.Sequential:
        """ Creates a convolutional layer from given config """

        module = nn.Sequential()

        self.filters = int(config["filters"])
        padding = int(config["pad"])
        kernel_size = int(config["size"])
        stride = int(config["stride"])

        if padding:
            pad = (kernel_size - 1) // 2
        else:
            pad = 0

        # Batch normalization
        try:
            batch_normalize = int(config["batch_normalize"])
            bias = False
        except:
            batch_normalize = 0
            bias = True

        # Add layer to module
        conv = nn.Conv2d(self.prev_filters, self.filters, kernel_size, stride, pad, bias = bias)
        module.add_module("conv_{0}".format(index), conv)

        if batch_normalize:
            bn = nn.BatchNorm2d(self.filters)
            module.add_module("batch_norm_{0}".format(index), bn)

        # Activation function
        activation = config["activation"]
        if activation == "leaky":
            activn = nn.LeakyReLU(0.1, inplace = True)
            module.add_module("leaky_{0}".format(index), activn)

        return module

    def create_upsample(self, index: int, config: Dict) -> nn.Sequential:
        """ Creates an upsampling layer from given config """

        module = nn.Sequential()

        stride = int(config["stride"])
        upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
        module.add_module("upsample_{}".format(index), upsample)

        return module

    def create_route(self, index: int, config: Dict) -> nn.Sequential:
        """ Creates a route layer from given config """

        module = nn.Sequential()

        config["layers"] = config["layers"].split(',')

        # Route start
        start = int(config["layers"][0])

        # Route end
        try:
            end = int(config["layers"][1])
        except:
            end = 0

        # Positive annotation
        if start > 0: 
            start = start - index
        if end > 0:
            end = end - index

        route = EmptyLayer()

        module.add_module("route_{0}".format(index), route)
        if end < 0:
            self.filters = self.output_filters[index + start] + self.output_filters[index + end]
        else:
            self.filters = self.output_filters[index + start]

        return module


    def create_shortcut(self, index: int, config: Dict) -> nn.Sequential():
        """ Creates a shortcut layer from given config """

        module = nn.Sequential()

        shortcut = EmptyLayer()
        module.add_module("shortcut_{}".format(index), shortcut)

        return module

    def create_yolo(self, index: int, config: Dict) -> nn.Sequential():
        """ Creates the YOLO layer from given config """

        module = nn.Sequential()

        mask = config["mask"].split(",")
        mask = [int(x) for x in mask]

        anchors = config["anchors"].split(",")
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
        anchors = [anchors[i] for i in mask]

        detection = DetectionLayer(anchors)
        module.add_module("Detection_{}".format(index), detection)

        return module
                                
    def create_modules(self) -> Tuple[Dict, nn.ModuleList]:
        """ Creates all modules """
   
        module_list = nn.ModuleList()
        self.prev_filters = 3
        self.output_filters = []
        
        # Create the modules
        for i, config in enumerate(self.blocks[1:]):

            module = self.create_layer[config["type"]](i, config)

            module_list.append(module)
            self.prev_filters = self.filters
            self.output_filters.append(self.filters)

        return self.blocks, module_list