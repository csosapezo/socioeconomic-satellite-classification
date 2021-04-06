import logging
import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import windows
from rasterio.features import geometry_mask

import utils.constants
import utils.get_labels

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def get_roof_segmentation_mask(labels_dict, image_shape, image_transform):
    """
    Generate a roof segmentation mask from a dictionary of geometries

    :param labels_dict: labels dictionary
    :type labels_dict: dict

    :param image_shape: satellite image patch shape
    :type image_shape: tuple

    :param image_transform: satellite image patch affine transform
    :type image_transform: any

    """

    geometries = []

    for _, labels in labels_dict.items():
        geometries += labels

    print(geometries)

    if geometries:
        mask = geometry_mask(geometries, image_shape, image_transform, all_touched=True, invert=True)
    else:
        mask = np.ndarray(image_shape)

    return mask


def get_income_level_segmentation_mask(labels_dict, levels, image_shape, image_transform):
    """
    Generate a roof segmentation mask from a dictionary of geometries

    :param labels_dict: labels dictionary
    :type labels_dict: dict

    :param levels: levels - mask index dictionary
    :type levels: dict

    :param image_shape: satellite image patch shape
    :type image_shape: tuple

    :param image_transform: satellite image patch affine transform
    :type image_transform: any

    """

    mask = np.ndarray((len(levels), image_shape[0], image_shape[1]))

    for level, labels in labels_dict.items():
        mask[levels[level]] = geometry_mask(labels, image_shape, image_transform) if labels else np.ndarray(image_shape)

    return mask


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


def split_images_and_generate_masks(images, database_path, output_path):
    """
        Create and save masks from geoJSON files.
        Implementation adapted from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies

        :param images: path in which satellite images are located
        :type images: str

        :param database_path: path in which the SQLite file is located
        :type database_path: str

        :param output_path: path in which mask are saved
        :type output_path: str
        """

    labels_output = os.path.join(output_path, "labels")
    split_output = os.path.join(output_path, "split")

    logger.debug("Mask output: {}".format(labels_output))
    logger.debug("Patches output: {}".format(split_output))

    if not os.path.exists(labels_output):
        os.mkdir(labels_output)

    if not os.path.exists(split_output):
        os.mkdir(split_output)

    levels = utils.get_levels(database_path)

    logger.debug("Income level names: {}".format(",".join(levels)))

    for image in images:

        image_basename = os.path.basename(image)
        dot = image_basename.rfind(".")
        output_basename = image_basename[:dot] + utils.constants.patch_suffix

        with rasterio.open(image) as dataset:

            meta = dataset.meta.copy()

            for window, transform in get_tiles(dataset):

                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height

                patch_output_filepath = os.path.join(split_output,
                                                     output_basename.format(int(window.col_off), int(window.row_off))
                                                     + utils.constants.dot_tif)
                logger.debug("Patch filename: {}".format(patch_output_filepath))

                labels_dict, num_labels = utils.get_labels(meta, image_basename, database_path)

                logger.debug("Amount of labels: {}".format(num_labels))

                roof_mask = get_roof_segmentation_mask(labels_dict, (meta['width'], meta['height']), meta['transform'])
                income_mask = get_income_level_segmentation_mask(labels_dict, levels,
                                                                 (meta['width'], meta['height']), meta['transform'])

                roof_mask_path = os.path.join(labels_output,
                                              output_basename.format(int(window.col_off), int(window.row_off))
                                              + utils.constants.roof_suffix
                                              + utils.constants.dot_npy)
                income_mask_path = os.path.join(labels_output,
                                                output_basename.format(int(window.col_off), int(window.row_off)) +
                                                utils.constants.income_suffix + utils.constants.dot_npy)

                with rasterio.open(patch_output_filepath, 'w', **meta) as outds:
                    patch_array = dataset.read(window=window)

                    if num_labels \
                            and (meta['width'] == utils.constants.width) and (meta['height'] == utils.constants.height):

                        pickle.dump(income_mask, open(str(income_mask_path), "wb"))
                        logger.debug("Income level mask saved at {}".format(str(income_mask_path)))

                        pickle.dump(roof_mask, open(str(roof_mask_path), "wb"))
                        logger.debug("Roof mask saved at {}".format(str(roof_mask_path)))

                        outds.write(patch_array)
                        logger.debug("Patch saved.")
