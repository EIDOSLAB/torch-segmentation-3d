from torch_segmentation_3d.base.encoder import BaseEncoder
from torch_segmentation_3d.encoders.resnet import resnet_encoders

__all_encoders__ = resnet_encoders


def get_encoder(name, in_channels, weights) -> BaseEncoder:
    encoder_fn = __all_encoders__[name]
    return encoder_fn(in_channels=in_channels, weights=weights)
