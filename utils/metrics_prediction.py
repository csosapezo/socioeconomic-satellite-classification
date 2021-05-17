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


def find_metrics(train_file_names, val_file_names, test_file_names, mask_dir, dataset, mean_values, std_values, model,
                 fold_out='0', fold_in='0', name_model='UNet11', epochs='40', out_file='VHR', dataset_file='VHR',
                 name_file='_VHR_60_fake'):
    outfile_path = 'predictions/{}/'.format(out_file)

    f = open(
        "predictions/{}/metric{}_{}_foldout{}_foldin{}_{}epochs.txt".format(out_file, name_file, name_model, fold_out,
                                                                            fold_in, epochs), "w+")
    f2 = open(
        "predictions/{}/pred_loss_test{}_{}_foldout{}_foldin{}_{}epochs.txt".format(out_file, name_file, name_model,
                                                                                    fold_out, fold_in, epochs), "w+")
    f.write("Training mean_values:[{}], std_values:[{}] \n".format(mean_values, std_values))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(len(test_file_names))

    all_transform = DualCompose([
        CenterCrop(int(dataset_file)),
        ImageOnly(Normalize(mean=mean_values, std=std_values))
    ])

    train_loader = utils.make_loader(filenames=train_file_names,
                                     mask_dir=mask_dir,
                                     dataset=dataset,
                                     shuffle=False,
                                     transform=all_transform,
                                     mode='train')

    val_loader = utils.make_loader(filenames=val_file_names,
                                   mask_dir=mask_dir,
                                   dataset=dataset,
                                   shuffle=False,
                                   transform=all_transform,
                                   mode='val')

    test_loader = utils.make_loader(filenames=test_file_names,
                                    mask_dir=mask_dir,
                                    dataset=dataset,
                                    shuffle=False,
                                    transform=all_transform,
                                    mode='test')

    dataloaders = {
        'train': train_loader, 'val': val_loader, 'test': test_loader}

    for phase in ['train', 'val', 'test']:
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

                if phase == 'test':
                    print_metrics(metrics, f2, 'test')

                pred = torch.sigmoid(pred)
                pred_vec.append(pred.data.cpu().numpy())

                result_dice += [metrics['dice']]

                result_jaccard += [metrics['jaccard']]

                count_img += 1

        print("{}_{}".format(phase, out_file))
        print('Dice = ', np.mean(result_dice), np.std(result_dice))
        print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard), '\n')

        f.write("{}_{}\n".format(phase, out_file))
        f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice), np.std(result_dice)))
        f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard), np.std(result_jaccard)))

        if phase == 'test':
            np.save(str(os.path.join(outfile_path,
                                     "inputs_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file, name_model,
                                                                                                  fold_out, fold_in,
                                                                                                  epochs,
                                                                                                  int(count_img)))),
                    np.array(input_vec))
            np.save(str(os.path.join(outfile_path,
                                     "labels_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file, name_model,
                                                                                                  fold_out, fold_in,
                                                                                                  epochs,
                                                                                                  int(count_img)))),
                    np.array(labels_vec))
            np.save(str(os.path.join(outfile_path,
                                     "pred_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file, name_model,
                                                                                                fold_out, fold_in,
                                                                                                epochs,
                                                                                                int(count_img)))),
                    np.array(pred_vec))
