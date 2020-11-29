import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models.modules.StackedHourGlass as M


class myUpsample(nn.Module):
	 def __init__(self):
		 super(myUpsample, self).__init__()
		 pass
	 def forward(self, x):
		 return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)


class Hourglass(nn.Module):
    """docstring for Hourglass"""
    def __init__(self, nChannels = 256, numReductions = 4, nModules = 2, poolKernel = (2,2), poolStride = (2,2), upSampleKernel = 2):
        super(Hourglass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride
        self.upSampleKernel = upSampleKernel
        """
        For the skip connection, a residual module (or sequence of residuaql modules)
        """
        # residual modules are used to pass the original image to subsequent modules
        # this is done to maintain the spatial relationship between features and solve
        # the vanishing gradient problem
        
        #upper branch
        _skip = []
        for _ in range(self.nModules):
            #add the residual modules to the network
            _skip.append(M.Residual(self.nChannels, self.nChannels, False))

        self.skip = nn.Sequential(*_skip)

        #lower branch
        """
        First pooling to go to smaller dimension then pass input through
        Residual Module or sequence of Modules then  and subsequent cases:
            either pass through Hourglass of numReductions-1
            or pass through M.Residual Module or sequence of Modules
        """
        # pool the image down to a lower resolution denoted by the window size of pool kernel
        self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

        _afterpool = []
        for _ in range(self.nModules):
            _afterpool.append(M.Residual(self.nChannels, self.nChannels, False))

        self.afterpool = nn.Sequential(*_afterpool)

        if (numReductions > 1):
            # recursively create downsampling layers within the hourglass
            self.hg = Hourglass(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
        else:
            #base case
            self.hg = M.Residual(self.nChannels,self.nChannels, False)


        """
        Now another M.Residual Module or sequence of M.Residual Modules
        """

        _lowres = []
        for _ in range(self.nModules):
            _lowres.append(M.Residual(self.nChannels,self.nChannels, False))

        self.lowres = nn.Sequential(*_lowres)

        # upsample
        self.up = nn.Upsample(scale_factor=upSampleKernel, mode='nearest')


    def forward(self, x):
        out1 = x
        out1 = self.skip(out1)
        out2 = x
        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        out2 = self.hg(out2)
        out2 = self.lowres(out2)
        out2 = self.up(out2)

        # return the input + the output to feed into the next hourglass
        return out2 + out1


class StackedHourGlass(nn.Module):
	"""docstring for StackedHourGlass"""
	def __init__(self, nChannels, nStack, nModules, numReductions, nJoints):
		super(StackedHourGlass, self).__init__()
		self.nChannels = nChannels # number of input chanels
		self.nStack = nStack # number of hourglasses to stack
		self.nModules = nModules # number of residual modules within the hourglass
		self.numReductions = numReductions # number of times the images are downsampled and upsampled
		self.nJoints = nJoints # number of joints to predict

        #set up pre processing
		self.start = M.BnReluConv(3, 64, kernelSize = 7, stride = 2, padding = 3)

		self.res1 = M.Residual(64, 128, False)
		self.mp = nn.MaxPool2d(2, 2)
		self.res2 = M.Residual(128, 128, False)
		self.res3 = M.Residual(128, self.nChannels, False)

		_hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]

        #create the stacks of hourglass networks
		for _ in range(self.nStack):
			_hourglass.append(Hourglass(self.nChannels, self.numReductions, self.nModules))
			_ResidualModules = []
			for _ in range(self.nModules):
				_ResidualModules.append(M.Residual(self.nChannels, self.nChannels, False))
			_ResidualModules = nn.Sequential(*_ResidualModules)
			_Residual.append(_ResidualModules)
			_lin1.append(M.BnReluConv(self.nChannels, self.nChannels))
			_chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints,1))
			_lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
			_jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

        # intermediate supervision at the end of the stack
		self.hourglass = nn.ModuleList(_hourglass)
		self.Residual = nn.ModuleList(_Residual)
		self.lin1 = nn.ModuleList(_lin1)
		self.chantojoints = nn.ModuleList(_chantojoints)
		self.lin2 = nn.ModuleList(_lin2)
		self.jointstochan = nn.ModuleList(_jointstochan)

	def forward(self, x):
        #preprocess the data
		x = self.start(x)
		x = self.res1(x)
		x = self.mp(x)
		x = self.res2(x)
		x = self.res3(x)
		out = []

        #feed forward the data through the network of stacked hourglasses
		for i in range(self.nStack):
			x1 = self.hourglass[i](x)
			x1 = self.Residual[i](x1)
            # make prediction
			x1 = self.lin1[i](x1)
			out.append(self.chantojoints[i](x1))
			x1 = self.lin2[i](x1)
			x = x + x1 + self.jointstochan[i](out[i])

		return (out)
