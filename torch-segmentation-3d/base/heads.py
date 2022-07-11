import torch.nn as nn

activations = {None: nn.Identity(), "none": nn.Identity(), "identity": nn.Identity(), "sigmoid": nn.Sigmoid()}


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(mode="trilinear", scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = activations[activation]
        super().__init__(conv3d, upsampling, activation)
