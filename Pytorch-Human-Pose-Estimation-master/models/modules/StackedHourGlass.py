import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BnReluConv(nn.Module):
		"""docstring for BnReluConv"""
		def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
				super(BnReluConv, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				self.kernelSize = kernelSize
				self.stride = stride
				self.padding = padding

				# apply batch normalization to input to improve training
				self.bn = nn.BatchNorm2d(self.inChannels)
				# apply convolution
				self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
				# after pooling a layer it is common practice to apply sigmoid or ReLU activation function to increase nonlinearity. 
				# using ReLU helps alleviate the vanishing gradient problem by having constent slope
				self.relu = nn.ReLU()

		def forward(self, x):
				x = self.bn(x)
				x = self.relu(x)
				x = self.conv(x)
				return x


class ConvBlock(nn.Module):
		"""docstring for ConvBlock"""
		def __init__(self, inChannels, outChannels):
				super(ConvBlock, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				self.outChannelsby2 = outChannels//2

				# bottlenecking restricts the total number of parameters at each layer 
				# curtailing total memory usage. Filters greater than 3x3 are never used
				self.cbr1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
				self.cbr2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
				self.cbr3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

		def forward(self, x):
				x = self.cbr1(x)
				x = self.cbr2(x)
				x = self.cbr3(x)
				return x

class SkipLayer(nn.Module):
		"""docstring for SkipLayer"""
		def __init__(self, inChannels, outChannels):
				super(SkipLayer, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				if (self.inChannels == self.outChannels):
						self.conv = None
				else:
						self.conv = nn.Conv2d(self.inChannels, self.outChannels, 1)

		def forward(self, x):
				if self.conv is not None:
						x = self.conv(x)
				return x

class Residual(nn.Module):
		"""docstring for Residual"""
		def __init__(self, inChannels, outChannels):
				super(Residual, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				self.cb = ConvBlock(inChannels, outChannels)
				self.skip = SkipLayer(inChannels, outChannels)

		def forward(self, x):
				out = 0
				#apply a convolution to the input
				out = out + self.cb(x)
				out = out + self.skip(x)
				return out


