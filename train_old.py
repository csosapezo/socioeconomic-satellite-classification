import argparse
import os

import torch
from torch.backends import cudnn

import models
import utils
from utils.metrics_prediction import find_metrics
from utils.transform import DualCompose, CenterCrop, HorizontalFlip, VerticalFlip, Rotate, ImageOnly, Normalize


def main():
    """
    Training function
    """

    # Argument parsing
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--images-path', type=str, default='./data/dataset/split/',
        help='path in which the satellite images are located')
    arg('--numpy-images-path', type=str, default='./data/train/split',
        help='path in which the satellite images will be located as numpy arrays')
    arg('--mask-path', type=str, default='./data/dataset/labels/',
        help='path in which labels are located as .npy files')
    arg('--data-path', type=str, default='./data/train/',
        help='path in which numpy arrays are stored as numPy arrays')
    arg('--batch-ssize', type=int, default=16, help='HR:4,VHR:8')
    arg('--lismit', type=int, default=10000, help='number of images in epoch')
    arg('--n-espochs', type=int, default=500)
    arg('--slr', type=float, default=1e-3)

    args = parser.parse_args()

    model = models.UNet()
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    cudnn.benchmark = True

    name_file = '_' + str(int(utils.constants.train_val_split * 100)) + '_percent'

    image_file_names = os.listdir(args.images_path)
    image_file_names.sort()
    print(f"Images: {len(image_file_names)}")

    labels_file_names = os.listdir(args.mask_path)
    labels_file_names.sort()
    print(f"Labels: {len(labels_file_names)}")

    train_set_indices, val_set_indices = utils.train_val_split(image_file_names, utils.constants.train_val_split)

    max_value, mean_train, std_train = utils.mean_std(args.images_path, image_file_names)

    np_file_names = utils.fill_zeros(args.images_path, image_file_names, args.numpy_images_path, mean_train)
    np_file_names.sort()

    print(f"Max value: {max_value}")
    print(f"Mean: {mean_train}")
    print(f"Std: {std_train}")

    train_transform = DualCompose([
        CenterCrop(utils.constants.height),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        ImageOnly(Normalize(mean=mean_train, std=std_train))
    ])

    val_transform = DualCompose([
        CenterCrop(utils.constants.height),
        ImageOnly(Normalize(mean=mean_train, std=std_train))
    ])

    train_set_images = [np_file_names[idx] for idx in train_set_indices]
    train_set_labels = [labels_file_names[idx] for idx in train_set_indices]

    val_set_images = [np_file_names[idx] for idx in val_set_indices]
    val_set_labels = [labels_file_names[idx] for idx in val_set_indices]

    train_loader = utils.make_loader(args.numpy_images_path, args.mask_path, train_set_images, train_set_labels,
                                     False, train_transform, 'train', args.batch_size)
    valid_loader = utils.make_loader(args.numpy_images_path, args.mask_path, val_set_images, val_set_labels,
                                     False, val_transform, 'train', args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    dataloaders = {
        'train': train_loader, 'val': valid_loader
    }

    utils.train_utils.train_model(
        name_file=name_file,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        name_model=model.__class__.__name__,
        num_epochs=args.n_epochs
    )

    if not os.path.exists(".model/"):
        os.mkdir(".model/")

    torch.save(model.module.state_dict(), '.model/model{}_{}_{}epochs.pth'.format(name_file, "UNet", args.n_epochs))

    find_metrics(images_path=args.numpy_images_path,
                 mask_path=args.mask_path,
                 train_set_images=train_set_images,
                 train_set_labels=train_set_labels,
                 val_set_images=val_set_images,
                 val_set_labels=val_set_labels,
                 mean_values=mean_train,
                 std_values=std_train,
                 model=model,
                 name_model="UNet",
                 epochs=args.n_epochs)


if __name__ == "__main__":
    main()
