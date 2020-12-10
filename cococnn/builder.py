import torch
import models
import losses
import metrics
import dataloaders as dataloaders


class Builder(object):
	"""docstring for Builder"""
	def __init__(self, opts):
		super(Builder, self).__init__()
		self.opts = opts
		if opts.loadModel is not None:
			self.states = torch.load(opts.loadModel)
		else:
			self.states = None
	def Model(self):
		ModelBuilder = getattr(models, self.opts.model)
		Model = ModelBuilder(self.opts.modelName, self.opts.hhKernel, self.opts.ohKernel, self.opts.nJoints)
		return Model

	def Loss(self):
		instance = losses.Loss(self.opts)
		return getattr(instance, self.opts.model)

	def Metric(self):
		PCKinstance = metrics.PCK(self.opts)
		return {'PCK' : getattr(PCKinstance, self.opts.model)}
			
	def Optimizer(self, Model):
		TrainableParams = filter(lambda p: p.requires_grad, Model.parameters())
		Optimizer = getattr(torch.optim, self.opts.optimizer_type)(TrainableParams, lr = self.opts.LR, eps = 1e-8)
		if self.states is not None and self.opts.loadOptim:
			Optimizer.load_state_dict(states['optimizer_state'])
			if self.opts.dropPreLoaded:
				for i,_ in enumarate(Optimizer.param_groups):
					Optimizer.param_groups[i]['lr'] /= opts.dropMagPreLoaded
		return Optimizer

	def DataLoaders(self):
		return dataloaders.ImageLoader(self.opts, 'train'), dataloaders.ImageLoader(self.opts, 'val')

	def Epoch(self):
		Epoch = 1
		if self.states is not None and self.opts.loadEpoch:
			Epoch = self.states['epoch']
		return Epoch
