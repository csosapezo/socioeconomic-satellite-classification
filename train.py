import argparse
import glob
import os

import torch
from torch.backends import cudnn
import numpy as np

import models
import utils
from utils.metrics_prediction import find_metrics
from utils.transform import DualCompose, CenterCrop, HorizontalFlip, VerticalFlip, Rotate, ImageOnly, Normalize


def train():
    """
    Training function
    Adapted from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies/
    """

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # image-related variables
    arg('--image-patches-dir', type=str, default='./data/dataset/split', help='satellite image patches directory')
    arg('--masks-dir', type=str, default='./data/dataset/labels', help='numPy masks directory')
    arg('--npy-dir', type=str, default='./data/dataset/split_npy', help='numPy preprocessed patches directory')
    arg('--train-dir', type=str, default='./data/train/', help='train sample directory')

    # preprocessing-related variables
    arg('--val-percent', type=float, default=0.25, help='Validation percent')
    arg('--test-percent', type=float, default=0.10, help='Test percent')

    # training-related variable
    arg('--batch-size', type=int, default=16, help='HR:4,VHR:8')
    arg('--limit', type=int, default=0, help='number of images in epoch')
    arg('--n-epochs', type=int, default=500)
    arg('--lr', type=float, default=1e-3)
    arg('--step', type=float, default=60)
    arg('--model', type=str, help='roof: roof segmentation / income: income determination')
    arg('--out-path', type=str, default='./trained_models/', help='model output path')
    arg('--pretrained', type=int, default=1, help='0: False; 1: True')

    # CUDA devices
    arg('--device-ids', type=str, default='0,1', help='For example 0,1 to run on two GPUs')

    args = parser.parse_args()

    pretrained = True if args.pretrained else False

    if args.model == "roof":
        model = models.UNet11(pretrained=pretrained)
    elif args.model == "income":
        model = models.UNet11(pretrained=pretrained, num_classes=2, input_channels=5)
    else:
        raise ValueError

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None

        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        cudnn.benchmark = True

    images_filenames = np.array(sorted(glob.glob(args.image_patches_dir + "/*.tif")))

    train_set_indices, val_set_indices, test_set_indices = utils.train_val_test_split(len(images_filenames),
                                                                                      args.val_percent,
                                                                                      args.test_percent)

    images_np_filenames = utils.save_npy(images_filenames, args.npy_dir, args.model, args.masks_dir)

    channel_num = 4 if args.model == "roof" else 5
    max_value, mean_train, std_train = utils.meanstd(np.array(images_np_filenames)[train_set_indices],
                                                     channel_num=channel_num)

    train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        ImageOnly(Normalize(mean=mean_train, std=std_train))
    ])

    val_transform = DualCompose([
        ImageOnly(Normalize(mean=mean_train, std=std_train))
    ])

    limit = args.limit if args.limit > 0 else None

    train_loader = utils.make_loader(filenames=np.array(images_np_filenames)[train_set_indices],
                                     mask_dir=args.masks_dir,
                                     dataset=args.model,
                                     shuffle=False,
                                     transform=train_transform,
                                     mode='train',
                                     batch_size=args.batch_size,
                                     limit=limit)

    valid_loader = utils.make_loader(filenames=np.array(images_np_filenames)[val_set_indices],
                                     mask_dir=args.masks_dir,
                                     dataset=args.model,
                                     shuffle=False,
                                     transform=val_transform,
                                     mode='train',
                                     batch_size=args.batch_size,
                                     limit=None)

    dataloaders = {
        'train': train_loader, 'val': valid_loader
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    name_file = '_' + str(int(args.val_percent * 100)) + '_percent_' + args.model

    utils.train_model(name_file=name_file,
                      model=model,
                      dataset=args.model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      dataloaders=dataloaders,
                      name_model="Unet11",
                      num_epochs=args.n_epochs)

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    torch.save(model.module.state_dict(),
               (str(args.out_path) + '/model{}_{}_{}epochs').format(name_file, "Unet11", args.n_epochs))

    find_metrics(train_file_names=np.array(images_np_filenames)[train_set_indices],
                 val_file_names=np.array(images_np_filenames)[val_set_indices],
                 test_file_names=np.array(images_np_filenames)[test_set_indices],
                 mask_dir=args.masks_dir,
                 dataset=args.model,
                 mean_values=mean_train,
                 std_values=std_train,
                 model=model,
                 name_model="Unet11",
                 epochs=args.n_epochs,
                 out_file=args.model,
                 dataset_file=args.model,
                 name_file=name_file)


if __name__ == "__main__":
    train()
