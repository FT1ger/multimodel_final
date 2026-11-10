"""Text cleaning and image transforms for CrisisMMD."""
import re

import ftfy
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def clean_text(text):
    """Fix encoding, remove URLs and retweet prefixes, mask mentions."""
    text = ftfy.fix_text(str(text))
    text = re.sub(r"^RT @\w+:\s*", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", "<user>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def eval_transform():
    """Deterministic image transform for feature extraction."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform():
    """Light augmentation transform for neural fusion training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path, missing, transform):
    """Return an image tensor and a missing flag."""
    if missing or not path:
        return torch.zeros(3, 224, 224), 1.0
    try:
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB"))
        return tensor, 0.0
    except (OSError, ValueError):
        return torch.zeros(3, 224, 224), 1.0


def class_weights(labels, label_order):
    """Return inverse-frequency class weights in label order."""
    counts = torch.zeros(len(label_order), dtype=torch.float64)
    index = {label: i for i, label in enumerate(label_order)}
    for label in labels:
        counts[index[label]] += 1
    weights = counts.sum() / (len(label_order) * counts.clamp(min=1))
    return weights.float()


def set_seed(seed):
    """Set Python, NumPy and Torch seeds for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
