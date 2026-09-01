import numpy as np
import torch
import torchvision.transforms.v2 as T

DEFAULT_WINDOWS = [[40, 80], [80, 200], [40, 380]]  # Brain, Subdural, Bone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def apply_window(image, center, width):
    """Applies Hounsfield Unit windowing and scales linearly to [0, 1]."""
    lower = center - width / 2.0
    w_image = (image.astype(np.float32) - lower) / width
    return np.clip(w_image, 0.0, 1.0)


class Transform:
    """Standard CT Transform: HU windowing -> Spatial Augs -> Normalization."""
    def __init__(self, train=False, image_size=224, windows=DEFAULT_WINDOWS):
        self.windows = windows
        if train:
            self.spatial = T.Compose([
                T.RandomResizedCrop(
                    (image_size, image_size),
                    scale=(0.9, 1.0),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=[-30, 30]),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        else:
            self.spatial = T.Resize(
                (image_size, image_size),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            )
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __repr__(self):
        return (f"Transform(\n"
                f"  windows={self.windows},\n"
                f"  spatial={self.spatial},\n"
                f"  normalize={self.normalize}\n"
                f")")

    def __call__(self, image, label=None):
        # 1. Step 1: HU Windowing -> 3 channels of shape (3, H, W)
        if isinstance(image, np.ndarray):
            channels = [apply_window(image, c, w) for c, w in self.windows]
            image = torch.from_numpy(np.stack(channels, axis=0)).float()
        image = self.spatial(image)
        image = self.normalize(image)
        if label is not None:
            label = torch.as_tensor(label).float()
            return image, label
        return image
