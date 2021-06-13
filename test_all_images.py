import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import UNet11
from show_sample_images import reverse_transform, masks_to_colorimg, pred_to_colorimg
from utils import preprocess_image
from utils.model_utils import load_model, run_model


def test_all_images():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # model-related variables
    arg('--model-path', type=str, help='Model path')
    arg('--dataset', type=str, help='roof: roof segmentation / income: income determination')

    # image-related variables
    arg('--masks-dir', type=str, default='./data/dataset/labels', help='numPy masks directory')
    arg('--npy-dir', type=str, default='./data/dataset/split_npy', help='numPy preprocessed patches directory')

    args = parser.parse_args()

    roof_path = "./data/dataset/split_npy"

    if args.dataset == "roof":
        model = load_model(args.model_path, UNet11)
    else:
        model = load_model(args.model_path, UNet11, input_channels=5, num_classes=4)
        # roof_model = load_model("./trained_models/model_10_percent_roof_Unet11_200epochs.pth", UNet11)

    if not os.path.exists("./data/test_all"):
        os.mkdir("./data/test_all")

    # Select sample pictures
    images_filenames = np.array(sorted(glob.glob(args.npy_dir + "/*.npy")))

    for filename in tqdm(images_filenames):

        fig = plt.figure(figsize=(10, 10))

        image = pickle.load(open(filename, "rb"))
        image = preprocess_image(image, args.dataset)

        pred = run_model(image, model, args.dataset)
        if args.dataset == "income":
            roof_image = pickle.load(open(os.path.join(roof_path, filename[filename.rfind("/") + 1:]), "rb"))
            roof_image = preprocess_image(roof_image, "roof")
            # pred_roof = run_model(roof_image, roof_model, "roof")
            # pred[0][0] = pred[0][0] * pred_roof[0][0]
            # pred[0][1] = pred[0][1] * pred_roof[0][0]

        mask_path = os.path.join(args.masks_dir, args.dataset, filename[filename.rfind("/") + 1:])
        y = pickle.load(open(mask_path, "rb"))

        fig.add_subplot(1, 3, 1)
        plt.imshow(reverse_transform(image.cpu().numpy()[0], args.dataset))

        fig.add_subplot(1, 3, 2)
        plt.imshow(masks_to_colorimg(y, args.dataset))

        fig.add_subplot(1, 3, 3)
        plt.imshow(pred_to_colorimg(pred.cpu().numpy(), args.dataset))

        plt.savefig(os.path.join("./data/test_all",
                                 filename[filename.rfind("/") + 1:filename.rfind(".")] + ".png"))

        plt.clf()
        plt.close(fig)


if __name__ == "__main__":
    test_all_images()
