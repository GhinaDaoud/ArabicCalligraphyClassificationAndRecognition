from __future__ import annotations
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(augmentation_cfg: dict | None = None) -> transforms.Compose:
    """Transforms for RGB inputs already resized and padded to the target canvas."""
    augmentation_cfg = augmentation_cfg or {}
    enable_online_aug = bool(augmentation_cfg.get("enable_online_aug", False))

    transform_steps: list = []
    if enable_online_aug:
        rotation = float(augmentation_cfg.get("rotation", 4.0))
        translate = float(augmentation_cfg.get("translate", 0.02))
        scale_min = float(augmentation_cfg.get("scale_min", 0.98))
        scale_max = float(augmentation_cfg.get("scale_max", 1.02))
        brightness = float(augmentation_cfg.get("brightness", 0.08))
        contrast = float(augmentation_cfg.get("contrast", 0.08))
        blur_prob = float(augmentation_cfg.get("blur_prob", 0.12))

        transform_steps.extend(
            [
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=rotation,
                            translate=(translate, translate),
                            scale=(scale_min, scale_max),
                            fill=255,
                        )
                    ],
                    p=0.6,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=brightness, contrast=contrast)],
                    p=0.45,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))],
                    p=blur_prob,
                ),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(transform_steps)

def build_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
