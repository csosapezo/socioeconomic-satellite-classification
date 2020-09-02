import os

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from rasterio.features import geometry_mask


def convert_geojson_to_numpy_array_mask(geojson_path, image_path):
    """
    Create a mask from a geoJSON file object.

    :param geojson_path: mask geoJSON path
    :type geojson_path: str

    :param image_path: satellite image path
    :type image_path: str


    :return: mask generated from geoJSON file
    :rtype: np.ndarray
    """

    with rasterio.open(image_path) as dataset:
        shape = (dataset.height, dataset.width)
        transform = dataset.transform

    with fiona.open(geojson_path) as f:
        geojson = [roof_data['geometry'] for roof_data in f]

    return geometry_mask(geojson, shape, transform, all_touched=True, invert=True)


def plot_mask(mask):
    """
    Plot a mask represented as a numpy array

    :param mask: numPy array
    :type mask: np.ndarray
    """

    plt.imshow(mask, cmap="gray")
    plt.show()


def generate_masks(image_directory_path, geojson_directory_path,
                   image_names, geojson_names, output_path):
    """
    Create and save masks from geoJSON files

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


    :return: a list containing mask names
    :rtype: list[str]
    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    mask_names = []

    for idx in range(len(image_names)):
        mask = convert_geojson_to_numpy_array_mask(str(os.path.join(geojson_directory_path, geojson_names[idx])),
                                                   str(os.path.join(image_directory_path, image_names[idx])))

        dot = image_names[idx].rfind(".") - 1
        name = image_names[idx][:dot] + ".npy"
        mask_names.append(name)

        np.save(str(os.path.join(output_path, "labels", name)), mask)

    return mask_names
