import argparse
import glob
import os
import pickle

import numpy as np


def fix_masks():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--masks-dir', type=str, default='./data/dataset/labels/income', help='numPy masks directory')
    arg('--npy-dir', type=str, default='./data/dataset/split_npy_income', help='numPy preprocessed patches directory')

    args = parser.parse_args()

    images_filenames = np.array(sorted(glob.glob(args.npy_dir + "/*.npy")))
    masks_dir = [os.path.join(args.masks_dir, filename[filename.rfind("/") + 1:]) for filename in images_filenames]

    for filename in masks_dir:
        mask = pickle.load(open(filename, "rb"))
        mask = mask.astype(bool).astype(int)
        assert mask.max() == 1
        assert mask.min() == 0
        print(filename)
        pickle.dump(mask, open(filename, "wb"))


if __name__ == "__main__":
    fix_masks()
