import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

masks_simple_path = "./data/dataset/labels/income"
outpath = './data/dataset/again_masks_fix_bug'

images = glob.glob(masks_simple_path + "/*")

for image in images:
    mask = pickle.load(open(image, "rb"))

    unified = mask[0] + mask[1]

    if unified.max() == 1 and mask.max() == 1 and mask.min() == 0:
        continue

    print(image, "max:", mask.max(), "unified_max:", unified.max())
    print("Capa 1 - min:", mask[0].min(), "max:", mask[0].max())
    print("Capa 2 - min:", mask[1].min(), "max:", mask[1].max())

    if mask[0].min() != 0 and mask[0].max() == 0:
        mask[0] = np.zeros((512, 512)).astype(int)
        pickle.dump(mask, open(image, "wb"))
    elif mask[1].min() != 0 and mask[1].max() == 0:
        mask[1] = np.zeros((512, 512)).astype(int)
        pickle.dump(mask, open(image, "wb"))

    # fig = plt.figure(figsize=(30, 10))

    # sp1 = fig.add_subplot(1, 3, 1)
    # fig.colorbar(plt.imshow(mask[0], vmin=0, vmax=1))

    # sp2 = fig.add_subplot(1, 3, 2)
    # fig.colorbar(plt.imshow(mask[1], vmin=0, vmax=1))

    # sp3 = fig.add_subplot(1, 3, 3)
    # fig.colorbar(plt.imshow(unified, vmin=0, vmax=1))

    # plt.savefig(os.path.join(outpath, image[image.rfind("/") + 1:image.rfind(".")] + ".png"))
    # plt.clf()
    # plt.close()
