import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from torch_segmentation_3d.encoders import get_encoder
from torch_segmentation_3d.base.heads import SegmentationHead
from torch_segmentation_3d.models.unet.decoder import UnetDecoder


class Unet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_exact=True,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        interp_output_size=False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            exact=decoder_exact
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.interp_output_size = interp_output_size
        self.name = "unet-{}".format(encoder_name)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        if self.interp_output_size:
            masks = F.interpolate(masks, size=x.shape[2:], mode="trilinear")

        return masks


if __name__ == "__main__":
    model = Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=3,
                 decoder_exact=False, interp_output_size=True)
    x = torch.randn((1, 1, 121, 145, 121))

    with torch.no_grad():
        print(model(x).shape)

    tot_params = 0
    for p in model.parameters():
        tot_params += p.numel()
    print("Total parameters:", tot_params)
