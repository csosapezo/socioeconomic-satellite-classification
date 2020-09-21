import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import utils
from utils import make_loader
from utils.loss import dice_loss, metric_jaccard  # this is loss
from utils.transform import DualCompose, CenterCrop, ImageOnly, Normalize


def calc_loss(pred, target, metrics, phase='train', bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)

    # convering tensor to numpy to remove from the computationl graph
    if phase == 'test':
        pred = (pred > 0.50).float()  # with 0.55 is a little better
        dice = dice_loss(pred, target)
        jaccard_loss = metric_jaccard(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)

        metrics['bce'] = bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] = loss.data.cpu().numpy() * target.size(0)
        metrics['dice'] = 1 - dice.data.cpu().numpy() * target.size(0)
        metrics['jaccard'] = 1 - jaccard_loss.data.cpu().numpy() * target.size(0)
    else:
        dice = dice_loss(pred, target)
        jaccard_loss = metric_jaccard(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        metrics['dice_loss'] += dice.data.cpu().numpy() * target.size(0)
        metrics['jaccard_loss'] += jaccard_loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, file, phase='train', epoch_samples=1):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    if phase == 'test':
        file.write("{}".format(",".join(outputs)))
    else:
        print("{}: {}".format(phase, ", ".join(outputs)))
        file.write("{}: {}".format(phase, ", ".join(outputs)))


def find_metrics(images_path, mask_path, train_set_images, train_set_labels, val_set_images, val_set_labels,
                 mean_values, std_values, model, name_model='UNet11', epochs=40):

    if not os.path.exists("predictions/"):
        os.mkdir("predictions/")

    f = open("predictions/metric_{}_{}_epochs.txt".format(name_model, epochs), "w+")
    f2 = open("predictions/pred_loss_test{}_{}_epochs.txt".format(name_model, epochs), "w+")
    f.write("Training mean_values:[{}], std_values:[{}] \n".format(mean_values, std_values))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_transform = DualCompose([
        CenterCrop(utils.constants.height),
        ImageOnly(Normalize(mean=mean_values, std=std_values))
    ])

    train_loader = make_loader(images_path, mask_path, train_set_images, train_set_labels, False, all_transform)
    val_loader = make_loader(images_path, mask_path, val_set_images, val_set_labels, False, all_transform)

    dataloaders = {
        'train': train_loader, 'val': val_loader}

    for phase in ['train', 'val']:
        model.eval()
        metrics = defaultdict(float)

        count_img = 0
        input_vec = []
        labels_vec = []
        pred_vec = []
        result_dice = []
        result_jaccard = []

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                input_vec.append(inputs.data.cpu().numpy())
                labels_vec.append(labels.data.cpu().numpy())
                pred = model(inputs)

                loss = calc_loss(pred, labels, metrics, 'test')

                print_metrics(metrics, f2, 'test')

                pred = torch.sigmoid(pred)
                pred_vec.append(pred.data.cpu().numpy())

                result_dice += [metrics['dice']]

                result_jaccard += [metrics['jaccard']]

                count_img += 1

        print(phase)
        print('Dice = ', np.mean(result_dice), np.std(result_dice))
        print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard), '\n')

        f.write(phase)
        f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice), np.std(result_dice)))
        f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard), np.std(result_jaccard)))
