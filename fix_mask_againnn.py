import glob
import os
import pickle

import rasterio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import get_labels, get_levels
from utils.generate_masks import get_roof_segmentation_mask, get_income_level_segmentation_mask

split_path = './data/dataset/split'
masks_complete_path = "./data/dataset/labels/income_complete"
masks_simple_path = "./data/dataset/labels/income"

images = glob.glob(split_path + "/*")
levels = get_levels("planos.sqlite")

for filename in tqdm(images):
    output_name = filename[filename.rfind("/") + 1:filename.rfind(".")] + ".png"
    output_name_npy = filename[filename.rfind("/") + 1:filename.rfind(".")] + ".npy"

    dataset = rasterio.open(filename)
    meta = dataset.profile

    labels_dict, _ = get_labels(dataset.profile, "IMG_PER1_20190217152904_ORT_P_000659.TIF", "planos.sqlite")
    roof_mask = get_roof_segmentation_mask(labels_dict, (meta['width'], meta['height']), meta['transform'])
    income_mask = get_income_level_segmentation_mask(labels_dict, levels,
                                                     (meta['width'], meta['height']), meta['transform'])

    pickle.dump(income_mask, open(os.path.join(masks_complete_path, output_name_npy), "wb"))
