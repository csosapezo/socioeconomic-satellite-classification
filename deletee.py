import os

masks_complete_path = "./data/dataset/labels/income_complete"
masks_simple_path = "./data/dataset/labels/income"
split_path = './data/dataset/split'
just_roof = "./data/dataset/split_just_roof"

os.mkdir(just_roof)

tif = ["IMG_PER1_20190217152904_ORT_P_000659_subtile_14336-7168.tif",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_18432-9216.tif",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_18432-9728.tif",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_19456-5120.tif",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_19968-10752.tif",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_21504-6656.tif",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_21504-7168.tif",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_21504-10752.tif",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_22016-10752.tif"
       ]

npy = ["IMG_PER1_20190217152904_ORT_P_000659_subtile_14336-7168.npy",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_18432-9216.npy",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_18432-9728.npy",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_19456-5120.npy",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_19968-10752.npy",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_21504-6656.npy",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_21504-7168.npy",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_21504-10752.npy",
       "IMG_PER1_20190217152904_ORT_P_000659_subtile_22016-10752.npy"
       ]

for tif_fie, npy_file in zip(tif, npy):
    os.remove(os.path.join(masks_complete_path, npy_file))
    os.remove(os.path.join(masks_simple_path, npy_file))
    os.rename(os.path.join(split_path, tif_fie), os.path.join(just_roof, tif_fie))
