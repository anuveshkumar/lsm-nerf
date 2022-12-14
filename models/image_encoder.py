import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImUnet(nn.Module):
    def __init__(self):
        super(ImUnet, self).__init__()
        # todo: No batch normalization yet, test with layer norm instead
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.deconv_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                           output_padding=(1, 1))
        self.deconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=2, padding=1,
                                           output_padding=(1, 1))
        self.deconv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1,
                                           output_padding=(1, 1))
        self.deconv_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=1)

        self.conv_bn1 = nn.BatchNorm2d(32)
        self.conv_bn2 = nn.BatchNorm2d(64)
        self.conv_bn3 = nn.BatchNorm2d(128)
        self.conv_bn4 = nn.BatchNorm2d(256)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv_bn3 = nn.BatchNorm2d(64)

    def forward(self, image):
        x_conv_1 = self.conv_1(image)
        x_conv_2 = self.conv_2(self.conv_bn1(F.relu(x_conv_1)))
        x_conv_3 = self.conv_3(self.conv_bn2(F.relu(x_conv_2)))
        x_conv_4 = self.conv_4(self.conv_bn3(F.relu(x_conv_3)))
        x_deconv_1 = self.deconv_1(self.conv_bn4(F.relu(x_conv_4)))
        x_deconv_1 = torch.cat((x_deconv_1, x_conv_3), dim=1)
        x_deconv_2 = self.deconv_2(self.deconv_bn1(F.relu(x_deconv_1)))
        x_deconv_2 = torch.cat((x_deconv_2, x_conv_2), dim=1)
        x_deconv_3 = self.deconv_3(self.deconv_bn2(F.relu(x_deconv_2)))
        x_deconv_3 = torch.cat((x_deconv_3, x_conv_1), dim=1)
        features = self.deconv_4(self.deconv_bn3(F.relu(x_deconv_3)))
        return features
