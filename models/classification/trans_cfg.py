from easydict import EasyDict as edict
import numpy as np

cfg = edict()

cfg.small = edict()
cfg.large = edict()

cfg.small.matrix = np.array([
    [1.25983807, -0.01639785, -14.61914208],
    [0.02008102,  1.19232813, -18.66017905],
])
cfg.small.res = (120, 160)

cfg.large.matrix = np.array([
    [1.26197198, -0.01742265, -55.42825743],
    [0.01858547,  1.19122363, -73.72868349],
])
cfg.large.res = (460, 640)