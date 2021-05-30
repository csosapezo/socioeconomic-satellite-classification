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
output_path = "./data/dataset/png_income_masks"
masks_complete_path = "./data/dataset/labels/income_complete"
masks_simple_path = "./data/dataset/labels/income"

os.mkdir(output_path)

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

    fig = plt.figure(figsize=(40, 10))

    simplified_mask = np.zeros((2, 512, 512)).astype(int)

    simplified_mask[0] = income_mask[levels["1.0"]] + income_mask[levels["2.0"]] + income_mask[levels["3.0"]]
    simplified_mask[1] = income_mask[levels["4.0"]] + income_mask[levels["5.0"]]

    mask = np.zeros((3, 512, 512)).astype(int)
    mask_s = np.zeros((3, 512, 512)).astype(int)

    coloring = [(50, 168, 82), (19, 125, 235), (241, 48, 254), (232, 138, 56), (178, 210, 209)]
    print("coloring")

    for idx, layer in enumerate(simplified_mask):
        mask_s[0] += layer * coloring[idx][0]
        mask_s[1] += layer * coloring[idx][1]
        mask_s[2] += layer * coloring[idx][2]

    for idx, layer in enumerate(income_mask):
        mask[0] += layer * coloring[idx][0]
        mask[1] += layer * coloring[idx][1]
        mask[2] += layer * coloring[idx][2]

    mask = mask.transpose((1, 2, 0))
    mask_s = mask_s.transpose((1, 2, 0))

    raster = dataset.read()
    raster_rgb = raster[:3]
    raster_rgb = (raster_rgb / 3512 * 255).astype(np.uint8)

    fig.add_subplot(1, 4, 1)
    plt.imshow(raster_rgb.transpose(1, 2, 0))

    fig.add_subplot(1, 4, 2)
    plt.imshow(roof_mask)

    fig.add_subplot(1, 4, 3)
    plt.imshow(mask)

    fig.add_subplot(1, 4, 4)
    plt.imshow(mask_s)

    plt.savefig(os.path.join(output_path, output_name))

    plt.clf()

    pickle.dump(income_mask, open(os.path.join(masks_complete_path, output_name_npy)))
    pickle.dump(simplified_mask, open(os.path.join(masks_simple_path, output_name_npy)))
