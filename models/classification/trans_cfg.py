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
