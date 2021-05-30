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
    plt.imshow(mask[0], vmin=0, vmax=unified.max())
    divider1 = make_axes_locatable(sp1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sp1, cax=cax1)

    sp2 = fig.add_subplot(1, 3, 2)
    plt.imshow(mask[1], vmin=0, vmax=unified.max())
    divider2 = make_axes_locatable(sp2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sp2, cax=cax2)

    sp3 = fig.add_subplot(1, 3, 3)
    plt.imshow(mask[1], vmin=0, vmax=unified.max())
    divider3 = make_axes_locatable(sp3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sp3, cax=cax3)

    plt.savefig(os.path.join(outpath, image[image.rfind("/") + 1:]))
    plt.clf()
    plt.close()
