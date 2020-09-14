import numpy as np
import torch
import os

from torch.utils.data.dataset import Dataset


class AOI_11_Rotterdam_Dataset(Dataset):
    def __init__(self, img_path, mask_path, img_names: list, mask_names: list,
                 transform=None, mode='train', limit=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_names = img_names
        self.mask_names = mask_names
        self.transform = transform
        self.mode = mode
        self.limit = limit

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_file_name = os.path.join(self.img_path, self.img_names[idx])
        mask_file_name = os.path.join(self.mask_path, self.mask_names[idx])

        img = load_image(img_file_name)

        if self.mode == 'train':
            mask = load_mask(mask_file_name)
            img, mask = self.transform(img, mask)
            mask = np.expand_dims(mask, -1)

            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            mask = np.zeros(img.shape[:2])
            img, mask = self.transform(img, mask)

            return to_float_tensor(img), str(img_file_name)


def to_float_tensor(img):
    img = torch.from_numpy(np.moveaxis(img, -1, 0)).float()
    return img


def load_image(path):  # Input:CH,H,W  Output:H,W,CH
    img = np.load(str(path), allow_pickle=True)
    img = img.transpose((1, 2, 0))
    return img


def load_mask(path):  # Input:H,W  Output:H,W,CH
    mask = np.load(path, allow_pickle=True)
    mask = np.expand_dims(mask, -1)
    mask = np.float32(mask)
    return mask
