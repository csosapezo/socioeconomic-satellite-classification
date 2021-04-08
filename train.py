import argparse
import glob

import torch
from torch.backends import cudnn
import numpy as np

import models
import utils
from utils.transform import DualCompose, CenterCrop, HorizontalFlip, VerticalFlip, Rotate, ImageOnly, Normalize


def train():
    """
    Training function
    Adapted from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies/
    """

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # image-related variables
    arg('--image-patches-dir', type=str, default='./data/dataset/split', help='satellite image patches directory')
    arg('--masks-dir', type=str, default='./data/dataset/labels', help='numPy masks directory')
    arg('--train-dir', type=str, default='./data/train/', help='train sample directory')

    # preprocessing-related variables
    arg('--val-percent', type=int, default=0.25, help='Validation percent')
    arg('--test-percent', type=int, default=0.10, help='Test percent')

    # training-related variable
    arg('--batch-size', type=int, default=16, help='HR:4,VHR:8')
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=500)
    arg('--lr', type=float, default=1e-3)
    arg('--model', type=str, help='ROOF: roof segmentation / INCOME: income determination')

    # CUDA devices
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')

    args = parser.parse_args()

    # TODO determine if different model architectures are needed for each task
    model = models.UNet()

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None

        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        cudnn.benchmark = True

    images_filenames = np.array(sorted(glob.glob(args.image_patches_dir + "/*.tif")))

    train_set_indices, val_set_indices, test_set_indices = utils.train_val_test_split(len(images_filenames),
                                                                                      args.val_percent,
                                                                                      args.test_percent)

    max_value, mean_train, std_train = utils.mean_std(images_filenames[train_set_indices])

    # TODO fill blanks on patches and save numPy files

    train_transform = DualCompose([
        CenterCrop(utils.constants.height),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        ImageOnly(Normalize(mean=mean_train, std=std_train))
    ])

    val_transform = DualCompose([
        CenterCrop(utils.constants.height),
        ImageOnly(Normalize(mean=mean_train, std=std_train))
    ])


if __name__ == "__main__":
    train()
