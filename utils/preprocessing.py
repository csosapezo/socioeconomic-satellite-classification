import glob
import os

import numpy as np
import rasterio
import pickle
import tifffile
import torch
import torchvision.transforms as T
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils.dataset import to_float_tensor
from utils.transform import DualCompose, CenterCrop, ImageOnly, Normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MeanStdDataset(Dataset):
    def __init__(self, files, tform=None, imgloader=tifffile.imread):
        # el contructor, se envia el dataset, trasfor, cargador de imagenes
        super(MeanStdDataset, self).__init__()

        self.filenames = files
        self.tform = tform
        self.imgloader = imgloader

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):  # retorna la imagen
        out = self.imgloader(self.filenames[i])
        out = out.astype(np.float32)  # lo vuelvo float32, int16 da errores
        if self.tform:
            out = self.tform(out)
        return out


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

            dot = name.rfind(".")
            np_name = name[:dot] + ".npy"
            np_image_names.append(np_name)

            pickle.dump(raster, open(str(os.path.join(output_path, "images", np_name)), "wb"))

    return np_image_names


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


def find_max(image_filenames):
    """
    Adapted from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies/
    """
    min_pixel = []
    max_pixel = []
    size = len(image_filenames)

    for filename in image_filenames:
        with rasterio.open(filename) as dataset:
            img = dataset.read()

            img = img.transpose((1, 2, 0))

            min_pixel.append(np.min(img))
            max_pixel.append(np.max(img))

    return np.min(min_pixel), np.max(max_pixel), size


def mean_std(image_filenames):
    """
    Implementation from Media_Desviacionstd_tifv2
    """
    min_pixel_all, max_pixel_all, size_all = find_max(image_filenames)

    doc_train_dataset = MeanStdDataset(files=image_filenames, tform=T.Compose([T.ToTensor()]))

    loader_med_sd = DataLoader(
        doc_train_dataset,
        batch_size=128,
        num_workers=0,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader_med_sd:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return max_pixel_all, mean.numpy(), std.numpy()


def fill_zeros(image_file_names, output_path, mean):
    np_image_names = []

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for name in image_file_names:
        with rasterio.open(name) as dataset:
            raster = dataset.read()

            raster[0][:][raster[0][:] == 0] = mean[0]
            raster[1][:][raster[0][:] == 0] = mean[1]
            raster[2][:][raster[0][:] == 0] = mean[2]
            raster[3][:][raster[0][:] == 0] = mean[3]

            dot = name.rfind(".")
            np_name = str(os.path.join(output_path, name[:dot] + ".npy"))
            np_image_names.append(np_name)

            pickle.dump(raster, open(np_name, "wb"))

    return np_image_names


def preprocess_image(img):
    """Normaliza y transforma la imagen en un tensor apto para ser procesado por la red neuronal de segmentación de
    cuerpos de agua.
    Dimensiones: entrada: (4,256,256); salida: (1,4,256,256)
    :param img: imagen por preprocesar
    :type img: np.ndarray
    """
    print(img.shape)
    img = img.transpose((1, 2, 0))
    image_transform = transform_function()
    img_for_model = image_transform(img)[0]
    img_for_model = Variable(to_float_tensor(img_for_model), requires_grad=False)
    img_for_model = img_for_model.unsqueeze(0).to(device)

    return img_for_model


def transform_function():
    """Función de normalización para una imagen satelital."""
    image_transform = DualCompose([CenterCrop(256), ImageOnly(
        Normalize(mean=[118.9308, 163.46594, 167.29922, 356.9652],
                  std=[75.99471, 83.11909, 102.464455, 202.43672]))])
    return image_transform
