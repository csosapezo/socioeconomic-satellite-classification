import argparse
import os

import utils


def main():
    """
    Function that creates a dataset
    """

    # Argument parsing
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--images', type=str, default='./data/train/',
        help='path in which the satellite images are located')
    arg('--database', type=str, default='./data/train/',
        help='path in which the label database is located')
    arg('--output', type=str, default='./data/train/',
        help='path in which numpy arrays (images and mask) are stored as numPy arrays')
    arg('--limit', type=int, default=100, help='maximum amount of images')

    args = parser.parse_args()

    # Preprocessing

    # List file names for rasters and geoJSON label data
    image_file_names = os.listdir(args.images)
    image_file_names.sort()

    if args.limit > 0:
        image_file_names = image_file_names[:args.limit]



    #utils.split_images_and_generate_masks(args.images, args.database, image_file_names, output)

if __name__ == "__main__":
    main()
