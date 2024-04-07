""" 
Based on https://github.com/qubvel/segmentation_models.pytorch
"""
import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):

        conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm)
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, exact=True):
        super().__init__()
        self.exact = exact
        self.conv1 = BasicBlock(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm
        )
        self.conv2 = BasicBlock(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            if skip.shape != x.shape and not self.exact:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = BasicBlock(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        conv2 = BasicBlock(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_batchnorm=True, center=False, exact=True):
        super().__init__()

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, exact=exact)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
