import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models.modules.PyraNet as M

'''
General Structure:
	1. Convolution layer
	2. Pyramid Residual Module + Pooling
	3. Pyramid Residual Module
	4. Hourglass stack 1 - score maps of body joint locations are produced at the end of 
		each hourglass, and a squared error loss is also attached in each stack of 
		hourglass
	.
	.
	.
	M. Hourglass stack n

General Hourglass Structure: hourglass network aims at capturing informatio at every scale 
	in feed-forward fashion (uses the proposed Pyramid Residual Module (PRMs) as a building 
	block)

	1. bottom up processing by subsampling the feature maps
	2. top down processing by upsampling the feature maps with the combination of higher 
		resolution features from bottom layers
	3. repeat for all iterations in the stack

	4. intermediate supervision at the end of the stack
		a. convolution layer
		b. score map
		c. convolution layer
		d. convolution layer

Pyramid Residual Module (PRM) Structure:
	- goal: learn feature pyramids across different levels of DCNNs (able to learn multi-scale 
	feature pyramids)

	- splits into two branches:
		1. standard convolution layers (no downsampling or upsampling)
		2. pyramid structure (convolution layers, including downsampling and upsampling steps)

	- PRM-A: produces separate input feature maps for different levels of pyramids
	- PRM-B: uses shared input for all levels of pyramids
	- PRM-c: use concatenation instead of addition to combine features generated from pyramids 
		(similar to inception models)
	- PRM-D: use dialated convolutions, instead of pooling to build the pyramid
'''


class PyraNetHourGlass(nn.Module):
    '''
        Hourglass Structure: constructs the stacked hourglass structure with PRMs as the building blocks
    '''

    def __init__(self, nChannels=256, numReductions=4, nModules=2, inputRes=256, baseWidth=6, cardinality=30, poolKernel=(2, 2), poolStride=(2, 2), upSampleKernel=2):
        super(PyraNetHourGlass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules  		# number of modules in each part of the hourglass structure
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride
        self.upSampleKernel = upSampleKernel

        self.inputRes = inputRes
        self.baseWidth = baseWidth
        self.cardinality = cardinality

        # use one Pyramid Residual Module (or multiple) to perform bottom up processing by subsampling
        #	the feature maps
        Residualskip = M.ResidualPyramid if numReductions > 1 else M.Residual
        Residualmain = M.ResidualPyramid if numReductions > 2 else M.Residual
        _skip = []
        for _ in range(self.nModules):
            _skip.append(Residualskip(self.nChannels, self.nChannels,
                                      self.inputRes, self.baseWidth, self.cardinality))

        self.skip = nn.Sequential(*_skip)

        """
		First pooling to go to smaller dimension then pass input through
		Residual Module or sequence of Modules then  and subsequent cases:
			either pass through Hourglass of numReductions-1
			or pass through Residual Module or sequence of Modules
		"""

        self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

        _afterpool = []
        for _ in range(self.nModules):
            _afterpool.append(Residualmain(
                self.nChannels, self.nChannels, self.inputRes//2, self.baseWidth, self.cardinality))

        self.afterpool = nn.Sequential(*_afterpool)

        if (numReductions > 1):
            self.hg = PyraNetHourGlass(self.nChannels, self.numReductions-1, self.nModules, self.inputRes //
                                       2, self.baseWidth, self.cardinality, self.poolKernel, self.poolStride, self.upSampleKernel)
        else:
            _num1res = []
            for _ in range(self.nModules):
                _num1res.append(Residualmain(
                    self.nChannels, self.nChannels, self.inputRes//2, self.baseWidth, self.cardinality))

            self.num1res = nn.Sequential(*_num1res)

        """
		Now another Residual Module or sequence of Residual Modules
		"""

        _lowres = []
        for _ in range(self.nModules):
            _lowres.append(Residualmain(self.nChannels, self.nChannels,
                                        self.inputRes//2, self.baseWidth, self.cardinality))

        self.lowres = nn.Sequential(*_lowres)

        # upsampling layer
        self.up = nn.Upsample(scale_factor=self.upSampleKernel)

    def forward(self, x):
        out1 = x
        out1 = self.skip(out1)
        out2 = x
        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        if self.numReductions > 1:
            out2 = self.hg(out2)
        else:
            out2 = self.num1res(out2)
        out2 = self.lowres(out2)
        out2 = self.up(out2)

        return out2 + out1


class PyraNet(nn.Module):
    '''
        Base class that structures the network as a whole
    '''

    def __init__(self, nChannels=256, nStack=4, nModules=2, numReductions=4, baseWidth=6, cardinality=30, nJoints=16, inputRes=256):
        super(PyraNet, self).__init__()
        self.nChannels = nChannels
        self.nStack = nStack
        self.nModules = nModules
        self.numReductions = numReductions
        self.baseWidth = baseWidth
        self.cardinality = cardinality
        self.inputRes = inputRes
        self.nJoints = nJoints

        # Convolution Layer
        self.start = M.BnReluConv(3, 64, kernelSize=7, stride=2, padding=3)

        # Pyramid Residual Module + Pooling
        self.res1 = M.ResidualPyramid(
            64, 128, self.inputRes//2, self.baseWidth, self.cardinality, 0)
        self.mp = nn.MaxPool2d(2, 2)

        # Pyramid Residual Module
        self.res2 = M.ResidualPyramid(
            128, 128, self.inputRes//4, self.baseWidth, self.cardinality,)

        # Pyramid Residual Module
        self.res3 = M.ResidualPyramid(
            128, self.nChannels, self.inputRes//4, self.baseWidth, self.cardinality)

        _hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [], [], [], [], [], []

        # Repeating Stacked Hourglass - repeat for n 'stacked hourglass' networks
        for _ in range(self.nStack):
            # Hourglass Structure using custom Pyramid Residual Modules (PRMs) as the building blocks
            _hourglass.append(PyraNetHourGlass(self.nChannels, self.numReductions,
                                               self.nModules, self.inputRes//4, self.baseWidth, self.cardinality))
            _ResidualModules = []
            for _ in range(self.nModules):
                _ResidualModules.append(M.Residual(
                    self.nChannels, self.nChannels))

            # intermediate supervision at the end of the stack
            _ResidualModules = nn.Sequential(*_ResidualModules)
            _Residual.append(_ResidualModules)
            _lin1.append(M.BnReluConv(self.nChannels, self.nChannels))
            _chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints, 1))
            _lin2.append(nn.Conv2d(self.nChannels, self.nChannels, 1))
            _jointstochan.append(nn.Conv2d(self.nJoints, self.nChannels, 1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin1 = nn.ModuleList(_lin1)
        self.chantojoints = nn.ModuleList(_chantojoints)
        self.lin2 = nn.ModuleList(_lin2)
        self.jointstochan = nn.ModuleList(_jointstochan)

    def forward(self, x):
        x = self.start(x)
        x = self.res1(x)
        x = self.mp(x)
        x = self.res2(x)
        x = self.res3(x)
        out = []

        for i in range(self.nStack):
            x1 = self.hourglass[i](x)
            x1 = self.Residual[i](x1)
            x1 = self.lin1[i](x1)
            out.append(self.chantojoints[i](x1))
            x1 = self.lin2[i](x1)
            x = x + x1 + self.jointstochan[i](out[i])

        return (out)
