"""
Based on the RetinaNet implementation of torchvision.
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
"""

import math
from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn

# from pathlib import Path
# import sys
# # parents = Path(__file__).resolve().parents
# # for p in parents:
#     # print(p)

# sys.path.append(str(Path(__file__).resolve().parents[6]))
# print(sys.path)


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
        # assert len(set(intermediate_features_dim)) == 1, "Feature dimensions of all stages have to same."
        in_channels = intermediate_features_dim[0]

        assert params.num_layers == len(
            params.in_channels
        ), "Number of layers and in_channels should be the same."

        # params = head.params

        anchors = params.anchors
        num_anchors = len(anchors[0]) // 2
        num_layers = len(anchors)

        self.num_layers = num_layers

        norm_type = params.norm_type
        use_act = False
        kernel_size = 1

        for i in range(num_layers):
            in_channels = params.in_channels[i]
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



        # TODO: Temporarily use hard-coded img_size
        self.anchor_generator = AnchorGenerator(
            params.anchors, aspect_ratios=[1.0], image_size=(480, 480)
        )
        # num_anchors = self.anchor_generator.num_anchors_per_location()[0]

        # self.classification_head = RetinaNetClassificationHead(
        #     in_channels, num_anchors, num_classes, num_layers, norm_layer=norm_layer
        # )
        # self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors, num_layers, norm_layer=norm_layer)

        # self.anchor_generator = AnchorGenerator(anchors, num_anchors, num_layers)

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


# from omegaconf import OmegaConf

# config = OmegaConf.load("config/model/yolo-fastest/yolo-fastest.yaml")
# cfg = config.model.architecture.head
# head = yolo_fastest_head(80, [256, 256], cfg)
# print(head)


# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     import sys
#     sys.path.append("../../../../../")
#     print(sys.path)

#     config = OmegaConf.load("config/model/yolo-fastest/yolo-fastest.yaml")
#     head = yolo_fastest_head(80, [256, 256], config.model.head)
#     print(head)
