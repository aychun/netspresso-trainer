# Benchmarks

We are working on creating pretrained weights with NetsPresso Trainer and our own resources. We base training recipes on the official repositories or original papers to replicate the performance of models.

For models that we have not yet trained with NetsPresso Trainer, we provide their pretrained weights from other awesome repositories. We have converted several models' weights into our own model architectures. We appreciate all the original authors and we also do our best to make other values.

Therefore, in the benchmark performance table of this section, a **Reproduced** status of True indicates performance obtained from our own training resources. In contrast, a False status means that the data is from original papers or repositories.

## Classification

| Dataset | Model | Weights | Resolution | Acc@1 | Acc@5 | Params | MACs | torch.fx | NetsPresso | Reproduced | Remarks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ImageNet1K | [EfficientFormer-l1](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/efficientformer/efficientformer-l1-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/efficientformer/efficientformer_l1_imagenet1k.safetensors?versionId=JIkKVaUF0fhkvLz2jfcY3MmbUg6MkUO6) | 224x224 | 80.20 | - | 12.30M | 1.30G | Supported | Supported | False | - |
| ImageNet1K | [MixNet-s](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/mixnet/mixnet-s-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_s_imagenet1k.safetensors?versionId=n0sHuieRyTWWzwBmSAE8oSP4BL53laDP) | 224x224 | 75.13 | - | - | - | Supported | Supported | False | - |
| ImageNet1K | [MixNet-m](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/mixnet/mixnet-m-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_m_imagenet1k.safetensors?versionId=cMkB57XAqu8Ro9OOWf9M6nLBPbrD2C7k) | 224x224 | 76.49 | - | - | - | Supported | Supported | False | - |
| ImageNet1K | [MixNet-l](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/mixnet/mixnet-l-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mixnet/mixnet_l_imagenet1k.safetensors?versionId=UZFlpK8LO_SlYbu5GnUe9Qb3srikM6mk) | 224x224 | 78.67 | - | - | - | Supported | Supported | False | - |
| ImageNet1K | [MobileNetV3-small](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/mobilenetv3/mobilenetv3-small-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilenetv3/mobilenet_v3_small_imagenet1k.safetensors?versionId=NTpIJOERdx4efzBgY7Wcca7Xe1_Vwal9) | 224x224 | 67.67 | 87.40 | 2.50M | 0.03G | Supported | Supported | False | - |
| ImageNet1K | [MobileViT](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/mobilevit/mobilevit-s-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/mobilevit/mobilevit_s_imagenet1k.safetensors?versionId=Kg71H367_VeSJqfzJv54At1uFcMyIf9D) | 224x224 | 78.40 | - | 5.60M | - | Supported | Supported | False | - |
| ImageNet1K | [ResNet18](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/resnet/resnet18-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet18_imagenet1k.safetensors?versionId=rI_BkIYyNFBtem180CSHA5QiGjuXgxMb) | 224x224 | 68.47 | 88.20 | 11.69M | 1.82G | Supported | Supported | True | - |
| ImageNet1K | [ResNet34](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/resnet/resnet34-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet34_imagenet1k.safetensors?versionId=YV687nYQc8tj5lq6ffqPpiJ8h2e0DW6L) | 224x224 | 72.26 | 90.63 | 21.80M | 3.67G | Supported | Supported | True | - |
| ImageNet1K | [ResNet50](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/resnet/resnet50-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/resnet/resnet50_imagenet1k.safetensors?versionId=kDZZabJz8kK.HWDtvo7VJ.HYZ7A3GcxS) | 224x224 | 79.61 | 94.67 | 25.56M | 2.62G | Supported | Supported | True | - |
| ImageNet1K | [ViT-tiny](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/vit/vit-tiny-classification.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/vit/vit_tiny_imagenet1k.safetensors?versionId=1WC4OqtnA5gJFolvCMrOWAdmiMwpL8RO) | 224x224 | 72.91 | - | 5.70M | - | Supported | Supported | False | - |

## Semantic segmentation

| Dataset | Model | Weights | Resolution | mIoU | Pixel acc | Params | MACs | torch.fx | NetsPresso | Reproduced | Remarks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| - | [SegFormer-b0](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/segformer/segformer-b0-segmentation.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/segformer/segformer_b0.safetensors?versionId=aZsJLrZrAysdvqRz2WVfCrjM.0sTFs3H) | - | - | - | - | - | Supported | Supported | False | - |
| Cityscapes | [PIDNet-s](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/pidnet/pidnet-s-segmentation.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/pidnet/pidnet_s_cityscapes.safetensors?versionId=lsgtDpiF1yqJpuCLYpruLdR6on0V53r8) | 2048x1024 | 78.8 | - | - | - | Supported | Supported | False | - |

## Object detection

| Dataset | Model | Weights | Resolution | mAP50 | mAP75 | mAP50:95 | Params | MACs | torch.fx | NetsPresso | Reproduced | Remarks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| COCO | [YOLOX-s](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/master/config/model/yolox/yolox-s-detection.yaml) | [download](https://netspresso-trainer-public.s3.ap-northeast-2.amazonaws.com/checkpoint/yolox/yolox_s_coco.safetensors?versionId=QRLqHKqhv8TSYBrmsQ3M8lCR8w7HEZyA) | 640x640 | 58.56 | 44.10 | 40.63 | 8.97M | 13.40G | Supported | Supported | True | conf_thresh=0.01, nms_thresh=0.65 |

## Acknowledgment

The original weight files which are not yet trained with NetsPresso Trainer are as follows.

- [EfficientFormer: apple/ml-cvnets](https://drive.google.com/file/d/11SbX-3cfqTOc247xKYubrAjBiUmr818y/view)
- [MobileViT: apple/ml-cvnets](https://apple.github.io/ml-cvnets/en/general/README-model-zoo.html#mobilevitv1-legacy)
- [ViT-tiny: apple/ml-cvnets](https://apple.github.io/ml-cvnets/en/general/README-model-zoo.html#classification-imagenet-1k)
- [SegFormer: (Hugging Face) nvidia](https://huggingface.co/nvidia/mit-b0) 
- [PIDNet: XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet#models)