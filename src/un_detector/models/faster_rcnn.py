# src/un_detector/models/faster_rcnn.py
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign


class BackboneWithChannels(nn.Module):
    """
    Custom backbone wrapper to return feature maps in a dict format
    compatible with Faster R-CNN.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return {'0': x}


def get_faster_rcnn_model(
    num_classes: int = 2,
    backbone_name: str = "resnet101",
    pretrained_backbone: bool = True,
) -> FasterRCNN:
    """
    Initialize Faster R-CNN with a ResNet-FPN backbone, optimized for hazard placard detection.

    Args:
        num_classes: Number of target classes including background (default: 2 -> background, hazmat).
        backbone_name: ResNet model backbone version (e.g. 'resnet101', 'resnet50').
        pretrained_backbone: Whether to load pre-trained weights for the backbone.

    Returns:
        Faster R-CNN model ready for training or inference.
    """
    # Create ResNet backbone with Feature Pyramid Network (FPN)
    backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained_backbone)

    # Define anchor generator for FPN
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Multi-scale RoI pooling for FPN
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3", "4"],
        output_size=7,
        sampling_ratio=2,
    )

    # Initialize Faster R-CNN with ResNet-101-FPN
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model
