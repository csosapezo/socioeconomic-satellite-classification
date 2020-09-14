import torch
import torch.nn as nn
from torch.nn import functional

"""
UNet PyTorch
Implementation adapted from https://github.com/usuyama/pytorch-unet
"""


def double_conv(in_channels, out_channels):
    """
    Double convolutional layer + ReLU activation

    :param in_channels: input channels
    :type in_channels: int

    :param out_channels: output channels
    :type out_channels: int

    :return: double convolutional layer (nn.Sequential)
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """
        Base UNet implementation
    """

    def __init__(self, input_channels=4, n_class=1):
        """
        Constructor method for UNet

        :param input_channels: input channels
        :param n_class: number of segmentation classes
        """
        self.n_class = n_class
        super().__init__()

        self.dconv_down1 = double_conv(input_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        """
        UNet Forward method
        :param x: input layer

        :return: segmented layer
        """

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        if self.n_class > 1:
            x = self.conv_last(x)
            out = functional.log_softmax(x, dim=1)
        else:
            out = self.conv_last(x)

        return out
