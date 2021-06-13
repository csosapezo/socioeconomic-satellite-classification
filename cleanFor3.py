import glob
import os
import pickle

import numpy as np

complete_dir = "./data/dataset/labels/income_complete"
two_dir = "./data/dataset/labels/income"

patches = glob.glob(two_dir + "/*")
new_patches = []

for filename in patches:
    if not os.path.exists("./data/dataset/labels/income_new"):
        os.mkdir("./data/dataset/labels/income_new")
    complete_fn = filename.replace("income", "income_complete")
    os.rename(complete_fn, complete_fn.replace("income_complete", "income_new"))
    new_patches.append(complete_fn.replace("income_complete", "income_new"))


for fn in new_patches:

    patch = pickle.load(open(fn, "rb"))
    patch = patch.astype(int)

    new_patch = np.zeros((4, 512, 512)).astype(int)

    for idx, layer in enumerate(patch):
        if layer.max() > 1:
            print(fn.rfind("/"), layer.max())
            layer[:][layer > 1] = 1
        if layer.min() < 0:
            print(fn.rfind("/"), layer.min())
            layer[:][layer < 0] = 0

    new_patch[0] = patch[1]
    new_patch[1] = patch[0]
    new_patch[2] = patch[3] + patch[4]

    if new_patch[2].max() > 1:
        new_patch[2][new_patch > 1] = 1

    new_patch[3] = np.logical_not(patch[0] + patch[1] + patch[3] + patch[4]).astype(int)

    pickle.dump(new_patch, open(fn, "wb"))
