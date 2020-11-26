import datasets
import torch
from torch.utils.data import DataLoader


def ImageLoader(opts, split):
    trainset = getattr(datasets, opts.dataset)(opts, split)
    evens = list(range(0, len(trainset), 2))
    trainset_1 = torch.utils.data.Subset(trainset, evens)

    return DataLoader(
        dataset=trainset_1,
        batch_size=opts.data_loader_size,
        shuffle=opts.shuffle if split == 'train' else False,
        pin_memory=not(opts.dont_pin_memory),
        num_workers=opts.nThreads
    )
