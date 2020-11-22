import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class BnReluConv(nn.Module):
    '''
		BnReluPyra: ensures that convolution is preceeded by:
			- batch normalization AND
			- ReLu

		Implementing these preceeding steps stops the variance explosion that occurrs when
		summing the outputs of the two residual units (outputs from each branch) in the PRM
    '''

    def __init__(self, inChannels, outChannels, kernelSize=1, stride=1, padding=0):
        super(BnReluConv, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding

        self.bn = nn.BatchNorm2d(self.inChannels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(self.inChannels, self.outChannels,
                              self.kernelSize, self.stride, self.padding)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class Pyramid(nn.Module):
	'''
		Pyramid: implements the unique (pyramid) part of the PRM
			1. downsample data
			2. convolution layer
			3. upsample data
			4. repeat steps 1-3 a designated number of times (given by cardinality)

		NOTE: this is only used in branch 2 of the PRM
	'''
    def __init__(self, D, cardinality, inputRes):
        super(Pyramid, self).__init__()
        self.D = D
        self.cardinality = cardinality
        self.inputRes = inputRes
        self.scale = 2**(-1/self.cardinality)
        _scales = []

		# repeating the downsampling, convolution, upsampling process a designated number
		# of times
        for card in range(self.cardinality):
            temp = nn.Sequential(
                nn.FractionalMaxPool2d(2, output_ratio=self.scale**(card + 1)),
                nn.Conv2d(self.D, self.D, 3, 1, 1),
                nn.Upsample(size=self.inputRes)  # , mode='bilinear')
            )
            _scales.append(temp)
        self.scales = nn.ModuleList(_scales)

    def forward(self, x):
        # print(x.shape, self.inputRes)
        out = torch.zeros_like(x)
        for card in range(self.cardinality):
            out += self.scales[card](x)
        return out


class BnReluPyra(nn.Module):
	'''
		BnReluPyra: ensures that when the Pyramid() class is called it is preceeded by:
			- batch normalization AND 
			- ReLu
			
		Implementing these preceeding steps stops the variance explosion that occurrs when
		summing the outputs of the two residual units (outputs from each branch) in the PRM
	'''
    def __init__(self, D, cardinality, inputRes):
        super(BnReluPyra, self).__init__()
        self.D = D
        self.cardinality = cardinality
        self.inputRes = inputRes

        self.bn = nn.BatchNorm2d(self.D)
        self.relu = nn.ReLU()
        self.pyra = Pyramid(self.D, self.cardinality, self.inputRes)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.pyra(x)
        return x


class ConvBlock(nn.Module):
    '''
        ConvBlock: class used to create a 'block' of conovolution layers

		NOTE: this is only used in branch 1 of the PRM
    '''

    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.outChannelsby2 = outChannels//2

		# sequence of 3 convolution layers (no downsampling or upsampling)

		# 1x1 convolution layer
        self.cbr1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
		# 3x3 convolution layer
        self.cbr2 = BnReluConv(self.outChannelsby2,
                               self.outChannelsby2, 3, 1, 1)
		# 1x1 convolution layer
        self.cbr3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        return x


class PyraConvBlock(nn.Module):
    '''
        PyraConvBlock: implements the inner (branched) portion of the PRM
    '''

    def __init__(self, inChannels, outChannels, inputRes, baseWidth, cardinality, type=1):
        super(PyraConvBlock, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.inputRes = inputRes
        self.baseWidth = baseWidth
        self.cardinality = cardinality
        self.outChannelsby2 = outChannels//2
        self.D = self.outChannels // self.baseWidth

        # branch 1 of PRM - series of convolution layers
        self.branch1 = nn.Sequential(
            BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0),
            BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
        )

        # branch 2 of PRM - incorporates convolution layers, downsampling & upsampling
        self.branch2 = nn.Sequential(
            BnReluConv(self.inChannels, self.D, 1, 1, 0),
            BnReluPyra(self.D, self.cardinality, self.inputRes),
            BnReluConv(self.D, self.outChannelsby2, 1, 1, 0)
        )

        # addition (occurrs at the end of all PRMs to unify branch results)
        self.afteradd = BnReluConv(
            self.outChannelsby2, self.outChannels, 1, 1, 0)

    def forward(self, x):
        x = self.branch2(x) + self.branch1(x)
        x = self.afteradd(x)
        return x


class SkipLayer(nn.Module):
    '''
        SkipLayer: implements the skipping of the subsampling and upsampling
    '''

    def __init__(self, inChannels, outChannels):
        super(SkipLayer, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels

		# doesn't try to perform convolution unless there is a difference between the number
		# of input layers and the number of output layers
        if (self.inChannels == self.outChannels):
            self.conv = None
        else:
            self.conv = nn.Conv2d(self.inChannels, self.outChannels, 1)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


class Residual(nn.Module):
    '''
        Residual: implements a residual block (with out the pyramid, no downsampling or upsampling, no branching)
    '''

    def __init__(self, inChannels, outChannels, inputRes=None, baseWidth=None, cardinality=None, type=None):
        super(Residual, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels

        # standard block of convolution layers (will skip the downsampling & upsampling steps)
        self.cb = ConvBlock(self.inChannels, self.outChannels)
        # skip layer
        self.skip = SkipLayer(self.inChannels, self.outChannels)

    def forward(self, x):
        out = 0
        out = out + self.cb(x)
        out = out + self.skip(x)
        return out


class ResidualPyramid(nn.Module):
    '''
        Residual Pyramid: class to implement a Pyramid Residual Module (PRM)
            - unlike the standard Residual() class, this class implements downsampling and upsampling
    '''

    def __init__(self, inChannels, outChannels, inputRes, baseWidth, cardinality, type=1):
        super(ResidualPyramid, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.inputRes = inputRes
        self.baseWidth = baseWidth
        self.cardinality = cardinality
        self.type = type

        # all types of Pyramid Residual Modules use a PyraConvBlock followed by a Skip Layer
        # the PyraConvBlock implements the branched portion of the PRM
        self.cb = PyraConvBlock(self.inChannels, self.outChannels,
                                self.inputRes, self.baseWidth, self.cardinality, self.type)
        self.skip = SkipLayer(self.inChannels, self.outChannels)

    def forward(self, x):
        out = 0
        out = out + self.cb(x)
        out = out + self.skip(x)
        return out
