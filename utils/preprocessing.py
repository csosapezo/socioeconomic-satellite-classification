import glob
import os

import numpy as np
import tifffile
import rasterio
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T

import utils


class MeanStdDataset(Dataset):
    def __init__(self, root, tform=None, imgloader=tifffile.imread):
        # el contructor, se envia el dataset, trasfor, cargador de imagenes
        super(MeanStdDataset, self).__init__()

        self.root = root
        self.filenames = sorted(glob.glob(root + "/*.tif"))
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


def mean_std(directory_path, image_filenames):
    """
    Implementation from Media_Desviacionstd_tifv2
    """
    min_pixel_all, max_pixel_all, size_all = find_max(directory_path, image_filenames)

    doc_train_dataset = MeanStdDataset(root=directory_path, tform=T.Compose([T.ToTensor()]))

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

    return max_pixel_all, mean, std


def fill_zeros(images_path, image_file_names, output_path, mean):
    np_image_names = []

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for name in image_file_names:
        with rasterio.open(str(os.path.join(images_path, name))) as dataset:
            raster = dataset.read()

            raster[0][:][raster[0][:] == 0] = mean[0]
            raster[1][:][raster[0][:] == 0] = mean[1]
            raster[2][:][raster[0][:] == 0] = mean[2]
            raster[3][:][raster[0][:] == 0] = mean[3]

            dot = name.rfind(".")
            np_name = name[:dot] + ".npy"
            np_image_names.append(np_name)

            np.save(str(os.path.join(output_path, np_name)), raster)

    return np_image_names
