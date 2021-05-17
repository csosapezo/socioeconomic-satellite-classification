import os

import torch
from torch.utils.data.dataloader import DataLoader

from utils.dataset import PeruSat1SegmentationDataset


def make_loader(filenames, mask_dir, dataset, shuffle=False, transform=None, mode='train', batch_size=4, limit=None):

    if dataset == "roof":
        return DataLoader(
            dataset=PeruSat1SegmentationDataset(filenames, str(os.path.join(mask_dir, dataset)), transform, mode,
                                                limit),
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )
    else:
        return None
