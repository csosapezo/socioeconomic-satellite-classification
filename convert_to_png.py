import argparse
import glob
import os

import numpy as np
import rasterio
from tqdm import tqdm


def convert_raster_to_png(filename, max_value, out_path):
    """Transforma una imagen satelital en una imagen png para su visualización en plataformas web.
    """

    basename = filename[filename.rfind("/") + 1:]

    dataset = rasterio.open(filename)
    raster = dataset.open()

    new_metadata = raster.profile
    new_metadata['count'] = 3
    new_metadata['driver'] = 'PNG'
    new_metadata['dtype'] = 'uint8'

    png_filename = basename[:basename.rfind(".")] + ".png"
    raster = raster[:3]
    new_raster = (raster / max_value * 255).astype('uint8')

    with rasterio.open(os.path.join(out_path, png_filename), 'w', **new_metadata) as dst:
        dst.write(new_raster)


def convert_to_png():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--image-patches-dir', type=str, default='./data/dataset/split', help='satellite image patches directory')
    arg('--png-patches-dir', type=str, default='./data/dataset/png_split', help='satellite image patches directory')
    arg('--max-pixel-value', type=int, default=3521)
    args = parser.parse_args()

    images_filenames = np.array(sorted(glob.glob(args.image_patches_dir + "/*.tif")))

    if not os.path.exists(args.png_patches_dir):
        os.mkdir(args.png_patches_dir)

    for filename in tqdm(images_filenames):

        convert_raster_to_png(filename, args.max_pixel_value, args.png_patches_dir)

