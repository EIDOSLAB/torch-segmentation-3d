from torch_segmentation_3d.encoders import resnet
from torch_segmentation_3d.base.encoder import BaseEncoder
from torch_segmentation_3d.encoders.resnet import resnet_encoders

__all_encoders__ = resnet_encoders


def get_encoder(name, in_channels, weights) -> BaseEncoder:
    if "resnet" in name:
        return resnet.build_encoder(name, weights, in_channels=in_channels)

    raise ValueError("Unkown encoder name")
