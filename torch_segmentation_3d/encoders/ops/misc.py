"""
Based on https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py
"""
from torch import nn


class SqueezeExcitation(nn.Module):

    def __init__(
            self,
            input_channels,
            squeeze_channels,
            activation=nn.ReLU,
            scale_activation=nn.Sigmoid,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv3d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv3d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input
