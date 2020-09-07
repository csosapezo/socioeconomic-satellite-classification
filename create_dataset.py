import argparse
import os

from torch.backends import cudnn

import utils


def main():
    """
    Function that creates a dataset
    """

    # Argument parsing
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--images-path', type=str, default='data/train/AOI_11_Rotterdam/PS-RGBNIR/',
        help='path in which the satellite images are located')
    arg('--geojson-path', type=str, default='data/train/AOI_11_Rotterdam/geojson_buildings/',
        help='path in which geoJSON labels are located')
    arg('--data-path', type=str, default='data/train/',
        help='path in which numpy arrays (images and mask) are stored as numPy arrays')
    arg('--geojson-name', type=str, default='SN6_Train_AOI_11_Rotterdam_Buildings_',
        help='geojson file name format')
    arg('--limit', type=int, default=100, help='maximum amount of images')

    args = parser.parse_args()

    cudnn.benchmark = True

    # Preprocessing

    # List file names for rasters and geoJSOM label data
    image_file_names = os.listdir(args.images_path)
    geojson_file_names = os.listdir(args.geojson_path)

    if args.limit > 0:
        image_file_names = image_file_names[:args.limit]
        geojson_file_names = geojson_file_names[:args.limit]

    utils.split_images_and_generate_masks(args.images_path, args.geojson_path,
                                          image_file_names, geojson_file_names, args.data_path)
