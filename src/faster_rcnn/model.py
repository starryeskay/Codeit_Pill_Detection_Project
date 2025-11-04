# model.py
"""
torchvision Faster R-CNN (ResNet50-FPN) + 헤드 교체로 파인튜닝
"""

import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights as W

def resolve_weights(arg: str):
    """
    --weights 인자를 Enum/None으로 변환
    """
    if arg is None:
        return W.DEFAULT
    name = str(arg).strip().lower()
    if name in ("default", "coco", "coco_v1"):
        return W.DEFAULT
    if name in ("none", "random", "rand"):
        return None
    # 호환용: 예전 문자열이 들어와도 DEFAULT로
    return W.DEFAULT

def build_model(num_classes: int,
                weights: str = "default",
                trainable_backbone_layers: int = 3,
                small_object_anchors: bool = True) -> nn.Module:

    wt = resolve_weights(weights)

    model = fasterrcnn_resnet50_fpn(
        weights=wt,
        trainable_backbone_layers=trainable_backbone_layers
    )
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=num_classes)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=num_classes*4)
    return model