import torch.nn as nn
import torch_segmentation_3d as seg3d


class EncoderCE(nn.Module):
    def __init__(self, encoder_name, n_classes):
        super().__init__()
        self.encoder = seg3d.encoders.get_encoder(encoder_name, in_channels=1, weights=None)

        feats = {"resnet18": 512}
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(feats[encoder_name], n_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features[-1])
