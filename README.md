3D Segmentations models for pytorch 

Supported architectures:
 - Unet
 - Unet++

Install:
  ```bash
  git clone git@github.com:EIDOSLAB/torch-segmentation-3d.git && cd torch-segmentation-3d
  pip3 install .
  ```

Example:
  ```python
  import torch_segmentation_3d as seg3d
  model = seg3d.models.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=1, classes=3)
  ```

Supported encoders:

- ResNet

  | Encoder  | Weights  | Params |
  |----------|----------|---------|
  | resnet18 | imagenet | 33M |
  | resnet34 | imagenet | 63M |
  | resnet50 | imagenet | 46M |
  | resnet101 | imagenet | 85M |

Cache location: 
- Linux: `~/.cache/torch/hub/checkpoints/resnet18-78569d5e.pth` (imagenet-1k)
