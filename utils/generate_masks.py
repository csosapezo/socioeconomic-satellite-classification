import os
import sqlite3
from itertools import product

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import pickle
from rasterio import windows
from pyproj import Transformer

from rasterio.features import geometry_mask

import utils.constants

"""def convert_geojson_to_numpy_array_mask(geojson_path, image_shape, image_transform):
    
    with fiona.open(geojson_path) as f:
        geojson = [roof_data['geometry'] for roof_data in f]

    if not geojson:
        return np.zeros(image_shape)

    return geometry_mask(geojson, image_shape, image_transform, all_touched=True, invert=True)"""


def plot_mask(mask):
    """
    Plot a mask represented as a numpy array

    :param mask: numPy array
    :type mask: np.ndarray
    """

    plt.imshow(mask, cmap="gray")
    plt.show()


def get_tiles(dataset, width=utils.constants.width, height=utils.constants.height):
    """
    Implementation from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies
    :param dataset: data reader
    :param width: patch expected width
    :param height: patch expected height
    :return: crop windows and Affine transform
    """
    nols, nrows = dataset.meta['width'], dataset.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, dataset.transform)
        yield window, transform


def split_images_and_generate_masks(image_directory_path, geojson_directory_path, image_names,
                                    geojson_names, output_path):
    """
        Create and save masks from geoJSON files.
        Implementation adapted from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies

        :param image_directory_path: path in which satellite images are located
        :type image_directory_path: str

        :param geojson_directory_path: path in which geojson files are located
        :type geojson_directory_path: str

        :param image_names: images names list
        :type image_names: list[str]

        :param geojson_names: geojson files names list
        :type geojson_names: list[str]

        :param output_path: path in which mask are saved
        :type output_path: str
        """

    # TODO 01: Update code to work with SQLite databases instead of geoJSONs

    for idx in range(len(image_names)):
        print("GeoJSON file: {}".format(geojson_names[idx]))
        geojson_filepath = os.path.join(geojson_directory_path, geojson_names[idx])
        image_filepath = os.path.join(image_directory_path, image_names[idx])

        if not os.path.exists(os.path.join(output_path, "labels")):
            os.mkdir(os.path.join(output_path, "labels"))

        if not os.path.exists(os.path.join(output_path, "split")):
            os.mkdir(os.path.join(output_path, "split"))

        dot = image_names[idx].rfind(".")
        output_filename = image_names[idx][:dot] + "_subtile_{}-{}.tif"

        with rasterio.open(image_filepath) as dataset:

            meta = dataset.meta.copy()

            for window, transform in get_tiles(dataset):
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height

                output_name = output_filename.format(int(window.col_off), int(window.row_off))
                patch_output_filepath = os.path.join(output_path, "split", output_name)

                # TODO 01.1: Replace convert_geojson_to_numpy_array_mask function
                # mask = convert_geojson_to_numpy_array_mask(geojson_filepath, (window.width, window.height),
                #                                           transform)
                # print(mask.max())
                dot = output_name.rfind(".")
                name = output_name[:dot] + ".npy"

                # print("Saved label: {}".format(os.path.join(output_path, "labels", name))

                with rasterio.open(patch_output_filepath, 'w', **meta) as outds:
                    # print("Saved patch: {}".format(patch_output_filepath))
                    patch_array = dataset.read(window=window)
                    sum_channels = np.sum(patch_array, axis=0)
                    equals0 = (sum_channels == 0).astype(np.uint8)
                    sum_percent = np.sum(equals0) / (window.width * window.height)

                    if sum_percent <= utils.constants.max_equals0 \
                            and (meta['width'] == utils.constants.width) and (meta['height'] == utils.constants.height):
                        outds.write(patch_array)
                        # pickle.dump(mask, open(str(os.path.join(output_path, "labels", name)), "wb"))
                    else:
                        os.remove(patch_output_filepath)
