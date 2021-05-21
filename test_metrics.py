import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from models import UNet11
from utils import preprocess_image
from utils.model_utils import load_model, run_model


def reverse_transform(inp):
    print(" image shape:", inp.shape)
    inp = inp.transpose(1, 2, 0)
    mean = np.array([0.09444648, 0.08571006, 0.10127277, 0.09419213])
    std = np.array([0.03668221, 0.0291096, 0.02894425, 0.03613606])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp / inp.max())
    inp = (inp * 255).astype(np.uint8)

    return inp


def masks_to_colorimg(mask):
    print("mask shape:", mask.shape)
    image = np.zeros(shape=[3, 512, 512])

    image[0] = mask[:] * 255
    image[1] = mask[:] * 255
    image[2] = mask[:] * 255

    image = image.transpose((1, 2, 0))
    image = image.astype('uint8')

    return image


def pred_to_colorimg(mask):
    print("pred shape:", mask.shape)
    mask = mask[0]
    for x in range(512):
        for y in range(512):
            mask[y, x] = (mask[0, y, x] > 0.5)

    image = masks_to_colorimg(mask)
    return image


def test_metrics():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # model-related variables
    arg('--model-path', type=str, help='Model path')
    arg('--num-picture', type=int, default=3, help='Number of sample pictures')
    arg('--dataset', type=str, help='roof: roof segmentation / income: income determination')

    # image-related variables
    arg('--image-patches-dir', type=str, default='./data/dataset/split', help='satellite image patches directory')
    arg('--masks-dir', type=str, default='./data/dataset/labels', help='numPy masks directory')
    arg('--npy-dir', type=str, default='./data/dataset/split_npy', help='numPy preprocessed patches directory')
    arg('--train-dir', type=str, default='./data/train/', help='train sample directory')

    args = parser.parse_args()

    modelname = args.model_path[args.model_path.rfind("/") + 1:args.model_path.rfind(".pth")]

    model = load_model(args.model_path, UNet11)

    print("Testing {} on {} samples".format(modelname, args.num_picture))

    # Select sample pictures
    images_filenames = np.array(sorted(glob.glob(args.npy_dir + "/*.npy")))
    sample_filenames = np.random.choice(images_filenames, args.num_picture)

    fig = plt.figure(figsize=(10, 10))

    for idx, filename in enumerate(sample_filenames):
        print("Loading sample input {}".format(idx))
        image = pickle.load(open(filename, "rb"))
        image = preprocess_image(image)

        print("Running model for sample {}".format(idx))
        pred = run_model(image, model)

        mask_path = os.path.join(args.masks_dir, args.dataset, filename[filename.rfind("/") + 1:])
        y = pickle.load(open(mask_path, "rb"))
        print("Get mask for sample {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 1)
        plt.imshow(reverse_transform(image.cpu().numpy()[0]))
        print("Add plot for sample input {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 2)
        plt.imshow(masks_to_colorimg(y))
        print("Add plot for sample mask {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 3)
        plt.imshow(masks_to_colorimg(pred.cpu().numpy()[0]))
        print("Add plot for sample pred {}".format(idx))

    if not os.path.exists("test"):
        os.mkdir("test")

    if not os.path.exists("test/{}".format(args.dataset)):
        os.mkdir("test/{}".format(args.dataset))

    plt.savefig("test/{}/test_{}_samples_{}.png"
                .format(args.dataset,
                        args.num_picture,
                        modelname))


if __name__ == "__main__":
    test_metrics()
