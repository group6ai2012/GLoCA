from __future__ import annotations

import random

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import torch
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224):
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_cdc_transforms(image_size: int = 224, augment_config: dict | None = None) -> dict[str, object]:
    augment_config = augment_config or {}
    strong_config = augment_config.get("strong", {}) if isinstance(augment_config, dict) else {}
    cutout_length = strong_config.get("cutout_length")
    if cutout_length is None:
        cutout_length = max(1, int(round(image_size * 0.25)))
    strong_steps = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size),
        RandAugmentLike(int(strong_config.get("randaugment_n", 4))),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if bool(strong_config.get("cutout_enabled", True)):
        strong_steps.append(
            Cutout(
                n_holes=int(strong_config.get("cutout_n_holes", 1)),
                length=int(cutout_length),
                random_length=bool(strong_config.get("cutout_random", True)),
            )
        )
    return {
        "weak": build_train_transform(image_size),
        "strong": transforms.Compose(strong_steps),
        "calibration": build_eval_transform(image_size),
    }


def _shear_x(img, value):
    if random.random() > 0.5:
        value = -value
    return img.transform(img.size, PIL.Image.AFFINE, (1, value, 0, 0, 1, 0))


def _shear_y(img, value):
    if random.random() > 0.5:
        value = -value
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, value, 1, 0))


def _translate_x(img, value):
    if random.random() > 0.5:
        value = -value
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, value * img.size[0], 0, 1, 0))


def _translate_y(img, value):
    if random.random() > 0.5:
        value = -value
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, value * img.size[1]))


def _rotate(img, value):
    if random.random() > 0.5:
        value = -value
    return img.rotate(value)


def _identity(img, _value):
    return img


def _autocontrast(img, _value):
    return PIL.ImageOps.autocontrast(img)


def _equalize(img, _value):
    return PIL.ImageOps.equalize(img)


def _solarize(img, value):
    return PIL.ImageOps.solarize(img, value)


def _posterize(img, value):
    return PIL.ImageOps.posterize(img, int(value))


def _contrast(img, value):
    return PIL.ImageEnhance.Contrast(img).enhance(value)


def _color(img, value):
    return PIL.ImageEnhance.Color(img).enhance(value)


def _brightness(img, value):
    return PIL.ImageEnhance.Brightness(img).enhance(value)


def _sharpness(img, value):
    return PIL.ImageEnhance.Sharpness(img).enhance(value)


def _cdc_augment_list():
    return [
        (_identity, 0.0, 1.0),
        (_autocontrast, 0.0, 1.0),
        (_equalize, 0.0, 1.0),
        (_rotate, -30.0, 30.0),
        (_solarize, 0.0, 256.0),
        (_color, 0.05, 0.95),
        (_contrast, 0.05, 0.95),
        (_brightness, 0.05, 0.95),
        (_sharpness, 0.05, 0.95),
        (_shear_x, -0.1, 0.1),
        (_translate_x, -0.1, 0.1),
        (_translate_y, -0.1, 0.1),
        (_posterize, 4.0, 8.0),
        (_shear_y, -0.1, 0.1),
    ]


class RandAugmentLike:
    def __init__(self, n: int) -> None:
        self.n = max(0, int(n))
        self.ops = _cdc_augment_list()

    def __call__(self, img):
        for op, low, high in random.choices(self.ops, k=self.n):
            value = random.random() * float(high - low) + float(low)
            img = op(img, value)
        return img


class Cutout:
    def __init__(self, n_holes: int, length: int, random_length: bool = True) -> None:
        self.n_holes = max(0, int(n_holes))
        self.length = max(1, int(length))
        self.random_length = bool(random_length)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.n_holes == 0:
            return tensor
        height = int(tensor.size(1))
        width = int(tensor.size(2))
        length = random.randint(1, self.length) if self.random_length else self.length
        mask = np.ones((height, width), dtype=np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(height)
            x = np.random.randint(width)
            y1 = np.clip(y - length // 2, 0, height)
            y2 = np.clip(y + length // 2, 0, height)
            x1 = np.clip(x - length // 2, 0, width)
            x2 = np.clip(x + length // 2, 0, width)
            mask[y1:y2, x1:x2] = 0.0
        return tensor * torch.from_numpy(mask).to(device=tensor.device, dtype=tensor.dtype).expand_as(tensor)
