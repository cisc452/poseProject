import torch
import torchvision
import torch.nn as nn  # https://pytorch.org/docs/stable/nn.html
import torch.nn.functional as F
import models.modules.DeepPose as M

'''
the DeepPose class extends the Pytorch Module (base class for all neural network modules)
modules can contain other modules, allowing to nest them in a tree structure (you can assign
	the submodules as regular attributes)
https://pytorch.org/docs/stable/generated/torch.nn.Module.html

Parameters:

self: self initialization

nJoints: number of joints the prediction will have

module Name: gives us the ability to use different models for our resnet
	all options are from https://arxiv.org/pdf/1512.03385.pdf
		- resnet18
		- resnet34 (default from config file)
		- resnet50 (default if param not given)
		- resnet101
		- resnet152
'''


class DeepPose(nn.Module):
    """docstring for DeepPose"""

    def __init__(self, nJoints, modelName='resnet50'):
        # initialize network as a Module
        # Params:	Model - DeepPose
        #			self - self
        super(DeepPose, self).__init__()

        # C (width x height x depth) - convolutional layer - contains learnable parameters
        # linear transformation followed by a non-linear transformation

        # LRN - local response normalizatio nlayer - parameter free

        # P - pooling layer - parameter free

        # F - fully connected layer - contains learnable parameters
        # linear transformation followed by a non-linear transformation

        # layer 1: 	C(55 × 55 × 96) − LRN − P
        # layer 2: 	C(27 × 27 × 256) − LRN − P
        # layer 3: 	C(13 × 13 × 384)
        # layer 4: 	C(13 × 13 × 384)
        # layer 5: 	C(13 × 13 × 256) − P
        # layer 6: 	F(4096)
        # layer 7: 	F(4096)

        # number of joints the prediction will have
        self.nJoints = nJoints

        # we add a bottleneck if we're using resnet > 34 (default is resnet50, config uses resnet34)
        self.block = 'BottleNeck' if (
            int(modelName[6:]) > 34) else 'BasicBlock'

        # use attributes from resnet50  (image classification model) to train the resnet
        # https://arxiv.org/abs/1512.03385

        # What is resnet?
        # https://missinglink.ai/guides/pytorch/pytorch-resnet-building-training-scaling-residual-networks-pytorch/
        # 	- CNN architecture that can support hundreds of convolutional layers (with no drop in effectiveness with
        # 		each additional layer)
        #	- proposes a solution to the 'vanishing gradient' problem that occurrs in backpropagation
        #		- identity shortcut connections - layer that initially don't do anything are added & initially skipped,
        # 			reusing the activatin functions from previous layers
        #		- when the network is trained again, the identical layers expand & help the network explore more of
        # 			the feature space
        # pretrained: specifies whether the model weights should be randomly initialized, or pre-trained on ImageNet
        self.resnet = getattr(torchvision.models, modelName)(pretrained=True)

        # applies a linear transformation to the incoming data
        # change the first layer to fit requirements
        self.resnet.fc = nn.Linear(
            512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)

    def forward(self, x):
        return self.resnet(x)
