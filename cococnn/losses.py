import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(object):
        """docstring for Loss"""
        def __init__(self, opts):
                super(Loss, self).__init__()
                self.opts = opts

        def ChainedPredictions(self, output, target, meta=None):
                return F.mse_loss(output*meta, target*meta)
