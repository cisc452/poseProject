import datasets
import torch
from torch.utils.data import DataLoader


def ImageLoader(opts, split):
    trainset = getattr(datasets, opts.dataset)(opts, split)
    # range(start, stop, step)
    numSamples = 1000
    mask = list(range(0, min(numSamples, len(trainset)), 1))
    trainset_1 = torch.utils.data.Subset(trainset, mask)

    return DataLoader(
        dataset=trainset_1,
        batch_size=opts.data_loader_size,
        shuffle=opts.shuffle if split == 'train' else False,
        pin_memory=not(opts.dont_pin_memory),
        num_workers=opts.nThreads
    )
