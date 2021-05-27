import argparse
import glob
import os
import pickle

import numpy as np
import rasterio

from utils import get_labels
from utils.constants import database_file
from utils.generate_masks import get_income_level_segmentation_mask

level_indices = [3, 2, 1, 4, 5]
levels_dict = {"3.0": 0, "2.0": 1, "1.0": 2, "4.0": 3, "5.0": 4}


def fix_masks(image_dir, masks_dir):
    images_filenames = np.array(sorted(glob.glob(image_dir + "/*.tif")))

    for filename in images_filenames:
        dataset = rasterio.open(filename)
        meta = dataset.profile
        labels_dict, num_labels = get_labels(meta, "IMG_PER1_20190217152904_ORT_P_000659.TIF", database_file)
        mask = get_income_level_segmentation_mask(labels_dict,
                                                  levels_dict, (meta['width'], meta['height']), meta['transform'])

        out_filename = os.path.join(masks_dir, filename[filename.rfind("/") + 1:])

        pickle.dump(mask, open(out_filename, "wb"))


def check_levels():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # image-related variables
    arg('--npy-dir', type=str, default='./data/dataset/split_npy', help='numPy preprocessed patches directory')
    arg('--fix', type=int, default=1, help="1: fix masks, 0: not")
    arg('--tif-dir', type=str, default='./data/dataset/split', help='patches directory')
    arg('--masks-dir', type=str, default='./data/dataset/labels/income', help='numPy masks directory')

    args = parser.parse_args()

    if args.fix != 1:
        fix_masks(args.tif_dir, args.masks_dir)
        return

    images_filenames = np.array(sorted(glob.glob(args.npy_dir + "/*.npy")))
    income_masks_filenames = \
        [os.path.join(args.masks_dir, filename[filename.rfind("/") + 1:]) for filename in images_filenames]

    num_layers_per_level = np.zeros(5).astype(int)

    for filename in income_masks_filenames:
        mask = pickle.load(open(filename, "rb"))

        for idx, level in enumerate(mask):
            if not np.isnan(level.max()):
                print(level.max())
                num_layers_per_level[idx] += int(level.max())
            else:
                level = np.zeros(level.shape)

        mask = mask.astype(int)

    print("Imager per layer:")

    for idx, level in enumerate(num_layers_per_level):
        print("Level {}: {}".format(level_indices[idx], level))


if __name__ == "__main__":
    check_levels()
