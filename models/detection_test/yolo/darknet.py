"""
YOLO v3 model
Based on https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

from loader import NetCreator
from util import predict_transform

def get_test_input(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (416,416))
    img =  img[:,:,::-1].transpose((2,0,1))
    img = img[np.newaxis,:,:,:] / 255.0
    img = torch.from_numpy(img).float()
    img = Variable(img)
    return img

class YOLOv3(nn.Module):
    """ YOLO v3 model based on darknet """

    def __init__(self, cfgfile):
        super(YOLOv3, self).__init__()
        self.blocks, self.module_list = NetCreator(cfgfile).create_modules()
        self.net_info = self.blocks[0]
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]

        # Cache route layer output
        outputs = {}
        
        write = 0
        for i, (config, module) in enumerate(zip(modules, self.module_list)):        
            
            if config["type"] in ["convolutional", "upsample"]:
                x = module(x)
    
            elif config["type"] == "route":
                layers = [int(layer) for layer in config["layers"]]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif config["type"] == "shortcut":
                from_ = int(config["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif config["type"] == 'yolo':        
                anchors = module[0].anchors

                # Get the input dimensions
                inp_dim = int(self.net_info["height"])
        
                # Get the number of classes
                num_classes = int(config["classes"])
        
                # Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
        
        return detections

