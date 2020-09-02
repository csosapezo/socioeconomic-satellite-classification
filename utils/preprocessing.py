import os

import numpy as np
import rasterio


def save_rasters_as_ndarrays(images_directory_path, image_names, output_path):
    """
    Save rasters located in a directory as numPy arrays on an specified location

    :param output_path: path in which arrays are saved
    :type output_path: str

    :param images_directory_path: path in which rasters are located
    :type images_directory_path: str

    :param image_names: list of image names
    :type image_names: list[str]

    :return: list of file names in which images are saved as arrays
    :rtpye: list[str]
    """

    np_image_names = []

    for name in image_names:
        with rasterio.open(str(os.path.join(images_directory_path, name))) as dataset:
            raster = dataset.read()

            dot = name.rfind(".") - 1
            np_name = name[:dot] + ".npy"
            np_image_names.append(np_name)

            np.save(str(os.path.join(output_path, "images", np_name)), raster)

    return np_image_names


def train_val_split(dataset, split_percent):
    """
    Split a dataset in train and validation subset

    :param dataset: List of elements of a dataset
    :type dataset: list

    :param split_percent: percentage of the dataset assigned to the validation subset
    :type split_percent: float


    :return: train and validation subset indices
    :rtype: (list[int], list[int])
    """

    dataset_size = len(dataset)

    indices = np.random.permutation(np.arange(dataset_size))
    val_size = split_percent * dataset_size

    train_set_indices = indices[val_size:]
    val_set_indices = indices[:val_size]

    return train_set_indices, val_set_indices
