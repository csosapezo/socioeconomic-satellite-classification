import argparse
import glob

import utils


def main():
    """
    Function that creates a dataset
    """

    # Argument parsing
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--images_dir', type=str, default='./data/images/pansharpen/',
        help='path in which the satellite images are located')
    arg('--database', type=str, default='./data/labels/planos.sqlite',
        help='path in which the label database is located')
    arg('--output', type=str, default='./data/dataset/',
        help='path in which numpy arrays (images and mask) are stored as numPy arrays')
    arg('--limit', type=int, default=100, help='maximum amount of images')

    args = parser.parse_args()

    # Preprocessing

    # List file names for rasters and geoJSON label data
    images = glob.glob(args.images_dir + "/*")

    if args.limit > 0:
        images = images[:args.limit]

    utils.split_images_and_generate_masks(images, args.database, args.output)


if __name__ == "__main__":
    main()
