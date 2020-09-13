import argparse
import os

from torch.backends import cudnn

import utils


def main():
    """
    Training function
    """

    # Argument parsing
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--images-path', type=str, default='./data/train/split/',
        help='path in which the satellite images are located')
    arg('--mask-path', type=str, default='./data/train/labels/',
        help='path in which labels are located as .npy files')
    arg('--data-path', type=str, default='./data/train/',
        help='path in which numpy arrays are stored as numPy arrays')
    arg('--batch-size', type=int, default=4, help='HR:4,VHR:8')
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=40)
    arg('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    cudnn.benchmark = True

    image_file_names = os.listdir(args.images_path)
    labels_file_names = os.listdir(args.mask_path)

    train_set_indices, val_set_indices = utils.train_val_split(image_file_names, utils.constants.train_val_split)

    mean_train, std_train = utils.mean_std(args.images_path, image_file_names, train_set_indices)




