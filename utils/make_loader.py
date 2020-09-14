import torch
from torch.utils.data.dataloader import DataLoader

from utils.dataset import AOI_11_Rotterdam_Dataset


def make_loader(img_path, mask_path, img_names, mask_names,
                shuffle=False, transform=None, mode='train', batch_size=4):
    return DataLoader(
        dataset=AOI_11_Rotterdam_Dataset(img_path, mask_path, img_names, mask_names, transform, mode),
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
