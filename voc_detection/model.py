from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes=21, pretrained=True):
    """Build the detector; disable pretrained downloads when loading a checkpoint."""
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT

    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights if pretrained else None,
        weights_backbone=None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model