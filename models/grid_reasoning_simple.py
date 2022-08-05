import torch
import torch.nn as nn
from inplace_abn import InPlaceABN


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=pad, bias=False)
        self.actvn = nn.ReLU()
        self.norm = nn.InstanceNorm3d(out_channels)
        self.bn = InPlaceABN(out_channels)

    def forward(self, x):
        return self.norm(self.actvn(self.conv(x)))


class Simple3DUNet(nn.Module):
    def __init__(self, in_channels, base_n_filter):
        super(Simple3DUNet, self).__init__()
        base_n_filter = base_n_filter
        self.conv0 = ConvBnReLU3D(in_channels, base_n_filter)

        self.conv1 = ConvBnReLU3D(base_n_filter, base_n_filter * 2, stride=2)
        self.conv2 = ConvBnReLU3D(base_n_filter * 2, base_n_filter * 2)

        self.conv3 = ConvBnReLU3D(base_n_filter * 2, base_n_filter * 4, stride=2)
        self.conv4 = ConvBnReLU3D(base_n_filter * 4, base_n_filter * 4)

        self.conv5 = ConvBnReLU3D(base_n_filter * 4, base_n_filter * 8, stride=2)
        self.conv6 = ConvBnReLU3D(base_n_filter * 8, base_n_filter * 8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_n_filter * 8, base_n_filter * 4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.ReLU(),
            nn.InstanceNorm3d(base_n_filter * 4),
            # InPlaceABN(base_n_filter * 4),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_n_filter * 4, base_n_filter * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.ReLU(),
            nn.InstanceNorm3d(base_n_filter * 1),
            # InPlaceABN(base_n_filter * 2),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_n_filter * 2, base_n_filter, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.ReLU(),
            nn.InstanceNorm3d(base_n_filter),
            # InPlaceABN(base_n_filter),
        )

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x
