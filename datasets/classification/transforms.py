import torch
from torchvision import transforms

from datasets.utils.augmentation.transforms import str_to_interp_mode, ToNumpy
from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def transforms_config(is_train: bool):
    transf_config = {
            'mean': IMAGENET_DEFAULT_MEAN,
            'std': IMAGENET_DEFAULT_STD
        }
    
    if is_train:
        transf_config.update({
            'hflip': 0.5,
            'vflip': 0,
        })

    return transf_config


def transforms_custom_train(
        img_size=64,
        hflip=0.5,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
):
    assert img_size > 32
    primary_tfl = [transforms.RandomResizedCrop(img_size, interpolation=str_to_interp_mode('bilinear')),
                   transforms.RandomHorizontalFlip(p=hflip),
                   transforms.ToTensor(),
                   transforms.Normalize(
                       mean=torch.tensor(mean),
                       std=torch.tensor(std)
                       )
    ]
    return transforms.Compose(primary_tfl)


def transforms_custom_eval(
        img_size=64,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        **kwargs):
    assert img_size > 32
    tfl = [
        transforms.Resize((img_size, img_size), str_to_interp_mode('bilinear')),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ]
    return transforms.Compose(tfl)


def create_classification_transform(args, is_training=False):
    return transforms_custom_train if is_training else transforms_custom_eval
