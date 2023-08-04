import os
from typing import Callable, Union, Optional
from abc import abstractmethod
import logging

import torch
import torch.nn as nn

from .registry import MODEL_BACKBONE_DICT, MODEL_HEAD_DICT
from .utils import BackboneOutput, ModelOutput, DetectionModelOutput, load_from_checkpoint

logger = logging.getLogger("netspresso_trainer")


class TaskModel(nn.Module):
    def __init__(self, conf_model, task, backbone_name, head_name, num_classes, model_checkpoint,
                 img_size: Optional[int] = None) -> None:
        super(TaskModel, self).__init__()
        self.task = task
        self.backbone_name = backbone_name
        self.head_name = head_name
        
        backbone_fn: Callable[..., nn.Module] = MODEL_BACKBONE_DICT[backbone_name]
        self.backbone: nn.Module = backbone_fn(task=self.task)
        
        self.backbone = load_from_checkpoint(self.backbone, model_checkpoint)
        
        head_module = MODEL_HEAD_DICT[self.task][head_name]
        label_size = img_size if task in ['segmentation', 'detection'] else None
        self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes, label_size=label_size)
        
    def _freeze_backbone(self):
        for m in self.backbone.parameters():
            m.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device
    
    def _get_name(self):
        return f"{self.__class__.__name__}[task={self.task}, backbone={self.backbone_name}, head={self.head_name}]"
    
    @abstractmethod
    def forward(self, x, label_size=None, targets=None):
        raise NotImplementedError


class ClassificationModel(TaskModel):
    def __init__(self, conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size=None) -> None:
        super().__init__(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size)
    
    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: ModelOutput = self.head(features['last_feature'])
        return out


class SegmentationModel(TaskModel):
    def __init__(self, conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size) -> None:
        super().__init__(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size)
    
    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: ModelOutput = self.head(features['intermediate_features'])
        return out


class DetectionModel(TaskModel):
    def __init__(self, conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size) -> None:
        super().__init__(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size)
    
    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: DetectionModelOutput = self.head(features['intermediate_features'], targets=targets)
        return out
    