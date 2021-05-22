import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from test_metrics import reverse_transform, masks_to_colorimg
from utils import preprocess_image


def check_masks():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # model-related variables
    arg('--dataset', type=str, help='roof: roof segmentation / income: income determination')

    # image-related variables
    arg('--masks-dir', type=str, default='./data/dataset/labels', help='numPy masks directory')
    arg('--npy-dir', type=str, default='./data/dataset/split_npy', help='numPy preprocessed patches directory')

    args = parser.parse_args()

    # Select sample pictures
    images_filenames = np.array(sorted(glob.glob(args.npy_dir + "/*.npy")))

    if not os.path.exists("/data/dataset/patch_plus_mask"):
        os.mkdir("/data/dataset/patch_plus_mask")

    for filename in images_filenames:

        fig = plt.figure(figsize=(10, 5))

        print("Loading patch {}".format(filename))
        image = pickle.load(open(filename, "rb"))
        image = preprocess_image(image)

        mask_path = os.path.join(args.masks_dir, args.dataset, filename[filename.rfind("/") + 1:])
        y = pickle.load(open(mask_path, "rb"))
        print("Get mask for {}".format(filename))

        fig.add_subplot(args.num_picture, 3, 1)
        plt.imshow(reverse_transform(image.cpu().numpy()[0]))
        print("Add plot for {}".format(filename))

        fig.add_subplot(args.num_picture, 3, 2)
        plt.imshow(masks_to_colorimg(y))
        print("Add plot for mask {}".format(filename))

        plt.savefig("./data/dataset/patch_plus_mask/{}.png".format(filename[filename.rfind("/") + 1:]))
        plt.clf()


if __name__ == "__main__":
    check_masks()
