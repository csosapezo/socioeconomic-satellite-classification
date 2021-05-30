import glob
import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

masks_simple_path = "./data/dataset/labels/income"
outpath = './data/dataset/again_masks_fix_bug'

images = glob.glob(masks_simple_path + "/*")

for image in images:
    mask = pickle.load(open(image, "rb"))

    unified = mask[0] + mask[1]

    fig = plt.figure(figsize=(30, 10))

    sp1 = fig.add_subplot(1, 3, 1)
    fig.colorbar(plt.imshow(mask[0], vmin=0, vmax=unified.max()))

    sp2 = fig.add_subplot(1, 3, 2)
    fig.colorbar(plt.imshow(mask[1], vmin=0, vmax=unified.max()))

    sp3 = fig.add_subplot(1, 3, 3)
    fig.colorbar(plt.imshow(mask[1], vmin=0, vmax=unified.max()))

    plt.savefig(os.path.join(outpath, image[image.rfind("/") + 1:image.rfind(".")] + ".png"))
    plt.clf()
    plt.close()
