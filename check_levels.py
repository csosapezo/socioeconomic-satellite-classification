import argparse
import glob
import os
import pickle

import numpy as np

level_indices = [3, 2, 1, 4, 5]


def check_levels():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # image-related variables
    arg('--npy-dir', type=str, default='./data/dataset/split_npy', help='numPy preprocessed patches directory')
    arg('--masks-dir', type=str, default='./data/dataset/labels/income', help='numPy masks directory')

    args = parser.parse_args()

    images_filenames = np.array(sorted(glob.glob(args.npy_dir + "/*.npy")))
    income_masks_filenames = \
        [os.path.join(args.masks_dir, filename[filename.rfind("/") + 1:]) for filename in images_filenames]

    num_layers_per_level = np.zeros(5).astype(int)

    for filename in income_masks_filenames:
        mask = pickle.load(open(filename, "rb"))

        for idx, level in enumerate(mask):
            if not np.isnan(level.max()):
                num_layers_per_level[idx] += int(level.max())
            else:
                level = np.zeros(level.shape)

        mask = mask.astype(int)

        pickle.dump(mask, open(filename, "wb"))

    print("Imager per layer:")

    for idx, level in enumerate(num_layers_per_level):
        print("Level {}: {}".format(level_indices[idx], level))


if __name__ == "__main__":
    check_levels()
