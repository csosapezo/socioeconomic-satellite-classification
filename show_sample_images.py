import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from models import UNet11
from utils import preprocess_image
from utils.model_utils import load_model, run_model


def reverse_transform(inp, dataset):
    print(" image shape:", inp.shape)
    inp = inp.transpose(1, 2, 0)
    if dataset == "roof":
        mean = np.array([0.09444648, 0.08571006, 0.10127277, 0.09419213])
        std = np.array([0.03668221, 0.0291096, 0.02894425, 0.03613606])
    else:
        mean = np.array([0.14308006, 0.12414238, 0.13847679, 0.14984046, 0.61647371])
        std = np.array([0.0537779,  0.04049726, 0.03915002, 0.0497247,  0.48624467])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp / inp.max())
    inp = (inp * 255).astype(np.uint8)
    inp = inp.transpose(2, 0, 1)
    inp = inp[:3]
    inp = inp.transpose(1, 2, 0)
    print(" image shape:", inp.shape)
    return inp


def masks_to_colorimg_roof(mask):
    print("mask shape:", mask.shape)
    image = np.zeros(shape=[3, 512, 512])

    image[0] = mask[:] * 255
    image[1] = mask[:] * 255
    image[2] = mask[:] * 255

    image = image.transpose((1, 2, 0))
    image = image.astype('uint8')

    return image


def mask_to_colorimg_income(mask):
    print("mask shape:", mask.shape)
    image = np.zeros(shape=[3, 512, 512])

    image[0] = mask[0] * 255
    image[1] = mask[1] * 255

    image = image.transpose((1, 2, 0))
    image = image.astype('uint8')

    return image


def masks_to_colorimg(mask, dataset):

    if dataset == "roof":
        return masks_to_colorimg_roof(mask)
    else:
        return mask_to_colorimg_income(mask)


def pred_to_colorimg_roof(mask):
    print("pred shape:", mask.shape)
    mask = mask[0]
    new_mask = np.zeros(shape=[512, 512])
    for x in range(512):
        for y in range(512):
            new_mask[y, x] = int(mask[0, y, x] > 0.5)

    image = masks_to_colorimg_roof(new_mask)
    return image


def pred_to_colorimg_income(mask):
    print("pred shape:", mask.shape)
    mask = mask[0]
    for x in range(512):
        for y in range(512):
            for z in range(2):
                mask[z, y, x] = int(mask[z, y, x] > 0.5)

    image = mask_to_colorimg_income(mask)
    return image


def pred_to_colorimg(mask, dataset):

    if dataset == "roof":
        return pred_to_colorimg_roof(mask)
    else:
        return pred_to_colorimg_income(mask)


def show_sample_images():
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

    roof_path = "./data/dataset/split"

    modelname = args.model_path[args.model_path.rfind("/") + 1:args.model_path.rfind(".pth")]

    if args.dataset == "roof":
        model = load_model(args.model_path, UNet11)
    else:
        model = load_model(args.model_path, UNet11, input_channels=5, num_classes=2)
        roof_model = load_model("./data/trained_models/model_10_percent_roof_Unet11_200epochs.pth", UNet11)

    print("Testing {} on {} samples".format(modelname, args.num_picture))

    # Select sample pictures
    images_filenames = np.array(sorted(glob.glob(args.npy_dir + "/*.npy")))
    sample_filenames = np.random.choice(images_filenames, args.num_picture)

    fig = plt.figure(figsize=(10, 10))

    for idx, filename in enumerate(sample_filenames):
        print("Loading sample input {}".format(idx))
        image = pickle.load(open(filename, "rb"))
        image = preprocess_image(image, args.dataset)

        print("Running model for sample {}".format(idx))
        pred = run_model(image, model, args.dataset)
        if args.dataset == "income":
            roof_image = pickle.load(open(os.path.join(roof_path, filename[filename.rfind("/") + 1:])))
            roof_image = preprocess_image(roof_image, "roof")
            pred_roof = run_model(roof_image, roof_model, "roof")
            pred[0][0] = pred[0][0] * pred_roof[0][0]
            pred[0][1] = pred[0][1] * pred_roof[0][0]

        mask_path = os.path.join(args.masks_dir, args.dataset, filename[filename.rfind("/") + 1:])
        y = pickle.load(open(mask_path, "rb"))
        print("Get mask for sample {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 1)
        plt.imshow(reverse_transform(image.cpu().numpy()[0], args.dataset))
        print("Add plot for sample input {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 2)
        plt.imshow(masks_to_colorimg(y, args.dataset))
        print("Add plot for sample mask {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 3)
        plt.imshow(pred_to_colorimg(pred.cpu().numpy(), args.dataset))
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
    show_sample_images()
