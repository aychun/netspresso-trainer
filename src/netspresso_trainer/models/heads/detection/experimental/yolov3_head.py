"""
Based on the RetinaNet implementation of torchvision.
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
"""

import math
from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn


from ....op.custom import ConvLayer
from ....utils import AnchorBasedDetectionModelOutput
from .detection import AnchorGenerator


class YoloFastestHead(nn.Module):
    
    num_layers: int
    def __init__(
        self,
        num_classes: int,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()

    

        anchors = params.anchors
        num_anchors = len(anchors[0]) // 2
        num_layers = len(anchors)

        self.num_layers = num_layers

        norm_type = params.norm_type
        use_act = False
        kernel_size = 1

        for i in range(num_layers):
            # in_channels = params.in_channels[i]

            in_channels = intermediate_features_dim[i]
            out_channels = params.out_channels[i]


            conv_norm = ConvLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                use_act=use_act,
            )
            conv = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                use_norm=False,
                use_act=use_act,
            )

            layer = nn.Sequential(conv_norm, conv)

            setattr(self, f"layer_{i+1}", layer)


        def init_bn(M):
            for m in M.modules():
                # print("###############")
                # print(m)
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.apply(init_bn)

        # print model

        print(" model head")
        print(self.layer_1)
        print(self.layer_2)
        exit()



    def forward(self, inputs: List[torch.Tensor]):
        # anchors = torch.cat(self.anchor_generator(x), dim=0)
        # cls_logits = self.classification_head(x)
        # bbox_regression = self.regression_head(x)

        x1, x2 = inputs
        out1 = self.layer_1(x1)
        out2 = self.layer_2(x2)


        print("IN YOLO FASTEST HEAD")
        print("Input shapes: ", x1.shape, x2.shape)
        print("Output shapes: ", out1.shape, out2.shape)


        return out1, out2

        # return AnchorBasedDetectionModelOutput(anchors=anchors, cls_logits=cls_logits, bbox_regression=bbox_regression)


def yolo_fastest_head(
    num_classes, intermediate_features_dim, conf_model_head, **kwargs
) -> YoloFastestHead:
    return YoloFastestHead(
        num_classes=num_classes,
        intermediate_features_dim=intermediate_features_dim,
        params=conf_model_head.params,
    )
