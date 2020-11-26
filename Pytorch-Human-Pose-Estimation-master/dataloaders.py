import datasets
from torch.utils.data import DataLoader


# class YourSampler(Sampler):
#     def __init__(self, mask):
#         self.mask = mask

#     def __iter__(self):
#         return (self.indices[i] for i in torch.nonzero(self.mask))

#     def __len__(self):
#         return len(self.mask)


# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)

# sampler1 = YourSampler(your_mask)
# sampler2 = YourSampler(your_other_mask)
# trainloader_sampler1 = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                                    sampler=sampler1, shuffle=False, num_workers=2)
# trainloader_sampler2 = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                                    sampler=sampler2, shuffle=False, num_workers=2)

# class sampler(Sampler):
# 	def __init__(self, mask):
# 		self.mask = mask

# 	def __iter__(self):
# 		return (self.indices[i] for i in torch.nonzero(self.mask))

# 	def __len__(self):
# 		return len(self.mask)

def ImageLoader(opts, split):
    return DataLoader(
        dataset=getattr(datasets, opts.dataset)(opts, split)[:1000],
        batch_size=opts.data_loader_size,
        shuffle=opts.shuffle if split == 'train' else False,
        pin_memory=not(opts.dont_pin_memory),
        num_workers=opts.nThreads
    )
