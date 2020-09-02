import argparse
import os

from torch.backends import cudnn

import utils


def main():
    """
    Main training function
    """

    # Argument parsing
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--percent', type=float, default=1, help='percent of data')
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=4, help='HR:4,VHR:8')
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=40)
    arg('--lr', type=float, default=1e-3)
    arg('--images-path', type=str, default='data/train/AOI_11_Rotterdam/PS-RGBNIR/',
        help='path in which the satellite images are located')
    arg('--geojson-path', type=str, default='data/train/AOI_11_Rotterdam/geojson_buildings/',
        help='path in which geoJSON labels are located')
    arg('--data-path', type=str, default='data/train/',
        help='path in which numpy arrays (images and mask) are stored as numPy arrays')
    arg('--geojson-name', type=str, default='SN6_Train_AOI_11_Rotterdam_Buildings_',
        help='geojson file name format')

    args = parser.parse_args()

    cudnn.benchmark = True

    # Preprocessing

    # List file names for rasters and geoJSOM label data
    image_file_names = os.listdir(args.images_path)
    geojson_file_names = os.listdir(args.geojson_path)

    mask_names = utils.generate_masks(args.images_path, args.geojson_path,
                                      image_file_names, geojson_file_names,
                                      args.data_path)

    np_image_names = utils.save_rasters_as_ndarrays(args.images_path, image_file_names, args.data_path)

    train_set_indices, val_set_indices = utils.train_val_split(image_file_names, args.percent)
