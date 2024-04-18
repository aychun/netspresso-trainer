from .common import CrossEntropyLoss, SigmoidFocalLoss
from .detection import RetinaNetLoss, YOLOXLoss, YoloFastestLoss
from .pose_estimation import RTMCCLoss
from .segmentation import PIDNetLoss

LOSS_DICT = {
    'cross_entropy': CrossEntropyLoss,
    'pidnet_loss': PIDNetLoss,
    'yolox_loss': YOLOXLoss,
    'retinanet_loss': RetinaNetLoss,
    'focal_loss': SigmoidFocalLoss,
    'rtmcc_loss': RTMCCLoss,
    'yolo_fastest_loss': YoloFastestLoss
}

PHASE_LIST = ['train', 'valid', 'test']
