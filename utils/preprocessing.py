import os
import pickle
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.autograd import Variable

from utils.dataset import to_float_tensor
from utils.transform import DualCompose, ImageOnly, Normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_val_test_split(dataset_size, val_percent, test_percent):
    """
    Split a dataset in train and validation subset

    :param dataset_size: dataset size
    :type dataset_size: int

    :param val_percent: percentage of the dataset assigned to the validation subset
    :type val_percent: float

    :param test_percent: percentage of the dataset assigned to the test subset
    :type test_percent: float


    :return: train and validation subset indices
    :rtype: (list[int], list[int], list[int])
    """

    np.random.seed(0)

    indices = np.random.permutation(np.arange(dataset_size))
    val_size = int(val_percent * dataset_size)
    test_size = int(test_percent * dataset_size)

    train_start = val_size + test_size

    val_set_indices = indices[:val_size]
    test_set_indices = indices[val_size:train_start]
    train_set_indices = indices[train_start:]

    return train_set_indices, val_set_indices, test_set_indices


def find_max(im_pths):
    """
        Implementation from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies/
    """
    min_pixel = []
    max_pixel = []
    size = len(im_pths)

    for i in im_pths:
        img = pickle.load(open(i, "rb"))

        img = img.transpose((1, 2, 0))

        min_pixel.append(np.min(img))
        max_pixel.append(np.max(img))

    return np.min(min_pixel), np.max(max_pixel), size


def cal_dir_stat(im_pths, max_value, channel_num):
    pixel_num = 0
    aux = None
    channel_sum = np.zeros(channel_num)
    channel_sum_squared = np.zeros(channel_num)

    for path in im_pths:
        im = pickle.load(open(path, "rb"))

        if channel_num != 4:
            aux = im[:][-1]

        im = im / max_value

        if channel_num != 4:
            im[-1] = aux

        im = im.transpose((1, 2, 0))
        pixel_num += (im.size / channel_num)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))

    return rgb_mean, rgb_std


def meanstd(root, rootdata='data_VHR', channel_num='4'):  # name_file,
    data_path = Path(rootdata)

    minimo_pixel_all, maximo_pixel_all, size_all = find_max(root)
    mean_all, std_all = cal_dir_stat(root, maximo_pixel_all, channel_num)

    print('All:', str(data_path), size_all, 'min ', np.min(minimo_pixel_all), 'max ', maximo_pixel_all)  # 0-1

    print("mean:{}\nstd:{}".format(mean_all, std_all))
    return maximo_pixel_all, mean_all, std_all


def save_npy(image_file_names, output_path, model, masks_dir):
    np_image_names = []

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for name in image_file_names:
        with rasterio.open(name) as dataset:
            raster = dataset.read()

            dot = name.rfind(".")
            slash = name.rfind("/") + 1

            if model == "income":
                roof = pickle.load(open(os.path.join(masks_dir, "roof", name[slash:dot] + ".npy"), "rb"))
                roof = np.float32(roof)
                raster = np.concatenate((raster, np.expand_dims(roof, 0)))

            np_name = str(os.path.join(output_path, name[slash:dot] + ".npy"))
            np_image_names.append(np_name)

            pickle.dump(raster, open(np_name, "wb"))

    return np_image_names


def preprocess_image(img, dataset):
    """Normaliza y transforma la imagen en un tensor apto para ser procesado por la red neuronal de segmentación de
    cuerpos de agua.
    Dimensiones: entrada: (4,256,256); salida: (1,4,256,256)
    :param img: imagen por preprocesar
    :type img: np.ndarray
    """
    print(img.shape)
    img = img.transpose((1, 2, 0))
    image_transform = transform_function(dataset)
    img_for_model = image_transform(img)[0]
    img_for_model = Variable(to_float_tensor(img_for_model), requires_grad=False)
    img_for_model = img_for_model.unsqueeze(0).to(device)

    return img_for_model


def transform_function(dataset):
    """Función de normalización para una imagen satelital."""
    if dataset == "roof":
        image_transform = DualCompose([ImageOnly(
            Normalize(mean=[0.09444648, 0.08571006, 0.10127277, 0.09419213],
                      std=[0.03668221, 0.0291096, 0.02894425, 0.03613606]))])
    else:
        image_transform = DualCompose([ImageOnly(
            Normalize(mean=[0.14308006, 0.12414238, 0.13847679, 0.14984046, 0.61647371],
                      std=[0.0537779,  0.04049726, 0.03915002, 0.0497247,  0.48624467]))])
    return image_transform



