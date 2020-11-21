import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models.modules.DeepPose as M


class DeepPose(nn.Module):
    """docstring for DeepPose"""

    def __init__(self, nJoints, modelName='resnet50'):
        # initialize network
        super(DeepPose, self).__init__()

        # C (width x height x depth) - convolutional layer - contains learnable parameters
        # linear transformation followed by a non-linear transformation

        # LRN - local response normalizatio nlayer - parameter free

        # P - pooling layer - parameter free

        # F - fully connected layer - contains learnable parameters
        # linear transformation followed by a non-linear transformation

        # First box (layer 1): 		C(55 × 55 × 96) − LRN − P
        # Second box (layer 2): 	C(27 × 27 × 256) − LRN − P
        # Third box (layer 3): 		C(13 × 13 × 384)
        # Fourth box (layer 4): 	C(13 × 13 × 384)
        # Fifth box (layer 5): 		C(13 × 13 × 256) − P
        # Sixth box (layer 6): 		F(4096)
        # Seventh box (layer 7): 	F(4096)

        # number of joints
        self.nJoints = nJoints

        self.block = 'BottleNeck' if (
            int(modelName[6:]) > 34) else 'BasicBlock'

        self.resnet = getattr(torchvision.models, modelName)(pretrained=True)

        self.resnet.fc = nn.Linear(
            512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)

    def forward(self, x):
        return self.resnet(x)
