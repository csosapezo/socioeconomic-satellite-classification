import os
import pickle

import numpy as np

masks_simple_path = "./data/dataset/labels/income"
outpath = './data/dataset/again_masks_fix_bug'

img1 = "IMG_PER1_20190217152904_ORT_P_000659_subtile_22528-5120.npy"
img2 = "IMG_PER1_20190217152904_ORT_P_000659_subtile_21504-4608.npy"
img3 = "IMG_PER1_20190217152904_ORT_P_000659_subtile_13824-8192.npy"

mask = pickle.load(open(os.path.join(masks_simple_path, img1), "rb"))
mask = mask.astype(bool).astype(int)

mask[1] = np.zeros((512, 512)).astype(int)

pickle.dump(mask, open(os.path.join(masks_simple_path, img1), "wb"))

mask = pickle.load(open(os.path.join(masks_simple_path, img2), "rb"))
mask = mask.astype(bool).astype(int)

mask[0] = np.zeros((512, 512)).astype(int)

pickle.dump(mask, open(os.path.join(masks_simple_path, img2), "wb"))


mask = pickle.load(open(os.path.join(masks_simple_path, img3), "rb"))
mask = mask.astype(bool).astype(int)

pickle.dump(mask, open(os.path.join(masks_simple_path, img3), "wb"))
