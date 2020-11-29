import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BnReluConv(nn.Module):
		"""batch normalizes applies 2d convolution then the relu activation"""
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

class DenselyConnectedConvBlock(nn.Module):
	def __init__(self, in_channels):    
		super(DenselyConnectedConvBlock, self).__init__()
		self.relu = nn.ReLU(inplace = True)
		self.bn = nn.BatchNorm2d(num_channels = in_channels)
		
		self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
		self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
		self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
		self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)  
		
	def forward(self, x):
		bn = self.bn(x) 
		conv1 = self.relu(self.conv1(bn))
		conv2 = self.relu(self.conv2(conv1))
		# Concatenate in channel dimension
		c2_dense = self.relu(torch.cat([conv1, conv2], 1))   
		conv3 = self.relu(self.conv3(c2_dense))
		c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

		conv4 = self.relu(self.conv4(c3_dense)) 
		c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

		conv5 = self.relu(self.conv5(c4_dense))
		c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

		return c5_dense

class ConvBlock(nn.Module):
		"""applies bottleneck convolution"""
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
		"""SkipLayer: implements the skipping of the subsampling and upsampling"""
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
		"""stacked hourglass residual modules consist of batch normalization followed by convolution and relu. the skip layer takes the input of this residual and passes it to the output of the residual"""
		def __init__(self, inChannels, outChannels, useDenseBlock):
				super(Residual, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				if (useDenseBlock):
					self.cb = DenselyConnectedConvBlock(inChannels)
				else:
					self.cb = ConvBlock(inChannels, outChannels)
				self.skip = SkipLayer(inChannels, outChannels)

		def forward(self, x):
				out = 0
				#apply a convolution to the input
				out = out + self.cb(x)
				out = out + self.skip(x)
				return out
