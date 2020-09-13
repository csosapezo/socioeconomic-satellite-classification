import os

import numpy as np
import rasterio

import utils


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
    val_size = int(split_percent * dataset_size)

    train_set_indices = indices[val_size:]
    val_set_indices = indices[:val_size]

    return train_set_indices, val_set_indices


def find_max(directory_path, image_filenames):
    """
    Adapted from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies/
    """
    min_pixel = []
    max_pixel = []
    size = len(image_filenames)

    for filename in image_filenames:
        with rasterio.open(str(os.path.join(directory_path, filename))) as dataset:
            img = dataset.read()

            img = img.transpose((1, 2, 0))

            min_pixel.append(np.min(img))
            max_pixel.append(np.max(img))

    return np.min(min_pixel), np.max(max_pixel), size


def cal_dir_stat(directory_path, image_filenames, set_indices=None,
                 maximun_value=utils.constants.pixel_max_value, num_channel=utils.constants.num_channel):
    """
    Adapted from https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
    """
    pixel_num = 0
    channel_sum = np.zeros(num_channel)
    channel_sum_squared = np.zeros(num_channel)

    if not set_indices:
        set_indices = list(range(len(image_filenames)))

    for idx in set_indices:
        with rasterio.open(str(os.path.join(directory_path, image_filenames[idx]))) as dataset:
            img = dataset.read()

            img = img / maximun_value
            pixel_num += (img.size / num_channel)
            channel_sum += np.sum(img, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(img), axis=(0, 1))

    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))

    return rgb_mean, rgb_std


def mean_std(directory_path, image_filenames, train_set_indices, num_channel=4):
    """
    Adapted from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies/
    """
    min_pixel_all, max_pixel_all, size_all = find_max(directory_path, image_filenames)

    mean_train, std_train = cal_dir_stat(directory_path, image_filenames, train_set_indices, max_pixel_all, num_channel)

    return mean_train, std_train
