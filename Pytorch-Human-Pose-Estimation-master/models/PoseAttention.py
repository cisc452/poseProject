import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models.modules.PoseAttention as M

class HourglassAttention(nn.Module):
	"""docstring for HourglassAttention"""
	def __init__(self, nChannels = 256, numReductions = 4, nModules = 2, poolKernel = (2,2), poolStride = (2,2), upSampleKernel = 2):
		super(HourglassAttention, self).__init__()
		self.numReductions = numReductions
		self.nModules = nModules
		self.nChannels = nChannels
		self.poolKernel = poolKernel
		self.poolStride = poolStride
		self.upSampleKernel = upSampleKernel
		"""
		For the skip connection, a Residual module (or sequence of residuaql modules)
		"""

		# Residual modules are added here
		# They are used to pass the original image at the initial resolution to other parts of the model
		_skip = []
		for _ in range(self.nModules):
			_skip.append(M.Residual(self.nChannels, self.nChannels))

		self.skip = nn.Sequential(*_skip)

		"""
		First pooling to go to smaller dimension then pass input through
		Residual Module or sequence of Modules then  and subsequent cases:
			either pass through Hourglass of numReductions-1
			or pass through Residual Module or sequence of Modules
		"""

		# Pool image
		# Downsampling to lower resolutions to find different features
		self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)
		_afterpool = []
		for _ in range(self.nModules):
			_afterpool.append(M.Residual(self.nChannels, self.nChannels))
		self.afterpool = nn.Sequential(*_afterpool)

		
		if (numReductions > 1):
			self.hg = HourglassAttention(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
		else:
			_num1res = []
			for _ in range(self.nModules):
				_num1res.append(M.Residual(self.nChannels,self.nChannels))

			self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

		"""
		Now another Residual Module or sequence of Residual Modules
		"""

		_lowres = []
		for _ in range(self.nModules):
			_lowres.append(M.Residual(self.nChannels,self.nChannels))

		self.lowres = nn.Sequential(*_lowres)

		# Upsampling
		# Downsample and then upsampling is where the"hourglass" gets its name 
		self.up = nn.Upsample(scale_factor = self.upSampleKernel)


	# Forward propagate
	# We need to return the initial input (x) along with the output
	# so we don't lose information
	def forward(self, x):
		out1 = x
		out1 = self.skip(out1)
		out2 = x
		out2 = self.mp(out2)
		out2 = self.afterpool(out2)
		if self.numReductions>1:
			out2 = self.hg(out2)
		else:
			out2 = self.num1res(out2)
		out2 = self.lowres(out2)
		out2 = self.up(out2)

		return out2 + out1


class PoseAttention(nn.Module):
	"""docstring for PoseAttention"""
	def __init__(self, nChannels, nStack, nModules, numReductions, nJoints, LRNSize, IterSize):
		super(PoseAttention, self).__init__()
		self.nChannels = nChannels			# Input channels (256 for coco)
		self.nStack = nStack				# Hourglass stacks (2 - Large effect of training time)
		self.nModules = nModules			# Residual hourglass units (2)
		self.numReductions = numReductions	# How many times the hourglasses down/upsample the images (4)
		self.nJoints = nJoints				# Number of joints we're predicting (16 for coco)
		self.LRNSize = LRNSize				# Local response Normalization layers (1)
		self.IterSize = IterSize			# How many times we're iterating through the attention part (3)

		self.start = M.BnReluConv(3, 64, kernelSize = 7, stride = 2, padding = 3)

		self.res1 = M.Residual(64, 128)
		self.mp = nn.MaxPool2d(2, 2)
		self.res2 = M.Residual(128, 128)
		self.res3 = M.Residual(128, self.nChannels)

		_hourglass, _Residual, _lin1, _attiter, _chantojoints, _lin2, _jointstochan = [], [],[],[],[],[],[]

		# Iterate through each hourglass in the stack
		for i in range(self.nStack):
			_hourglass.append(HourglassAttention(self.nChannels, self.numReductions, self.nModules))
			_ResidualModules = []
			for _ in range(self.nModules):
				_ResidualModules.append(M.Residual(self.nChannels, self.nChannels))
			_ResidualModules = nn.Sequential(*_ResidualModules)
			_Residual.append(_ResidualModules)
			_lin1.append(M.BnReluConv(self.nChannels, self.nChannels))
			_attiter.append(M.AttentionIter(self.nChannels, self.LRNSize, self.IterSize))

			# Add Sequential for lower half of stacks, CRF for upper half
			if i<self.nStack//2:
				_chantojoints.append(
						nn.Sequential(
							nn.BatchNorm2d(self.nChannels),
							nn.Conv2d(self.nChannels, self.nJoints,1),
						)
					)
			else:
				_chantojoints.append(M.AttentionPartsCRF(self.nChannels, self.LRNSize, self.IterSize, self.nJoints))
			_lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
			_jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

		self.hourglass = nn.ModuleList(_hourglass)
		self.Residual = nn.ModuleList(_Residual)
		self.lin1 = nn.ModuleList(_lin1)
		self.chantojoints = nn.ModuleList(_chantojoints)
		self.lin2 = nn.ModuleList(_lin2)
		self.jointstochan = nn.ModuleList(_jointstochan)

	# Forward propagate
	# Here, we can just return the network's output
	def forward(self, x):
		x = self.start(x)
		x = self.res1(x)
		#print("1", x.mean())
		x = self.mp(x)
		x = self.res2(x)
		#print("2", x.mean())
		x = self.res3(x)
		out = []

		for i in range(self.nStack):
			#print("3", x.mean())
			x1 = self.hourglass[i](x)
			#print("4", x1.mean())
			x1 = self.Residual[i](x1)
			#print("5", x1.mean())
			x1 = self.lin1[i](x1)
			#print("6", x1.mean())
			out.append(self.chantojoints[i](x1))
			x1 = self.lin2[i](x1)
			#print("7", x1.mean())
			x = x + x1 + self.jointstochan[i](out[i])

		return (out)
