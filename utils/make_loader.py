import os

import torch
from torch.utils.data.dataloader import DataLoader

from utils.dataset import PeruSat1SegmentationDataset


def make_loader(filenames, mask_dir, dataset, shuffle=False, transform=None, mode='train', batch_size=4, limit=None):
    return DataLoader(
            dataset=PeruSat1SegmentationDataset(filenames=filenames,
                                                mask_dir=str(os.path.join(mask_dir, dataset)),
                                                dataset=dataset,
                                                transform=transform,
                                                mode=mode,
                                                limit=limit),
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )
