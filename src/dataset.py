"""Datasets for pooled and token-level CrisisMMD features."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data import LABELS

FINAL = Path(__file__).resolve().parents[1]
CACHE = FINAL / "cache"


def load_index(path, key):
    """Return a dict from string id to row index and the feature array."""
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data[key]]
    return {value: i for i, value in enumerate(ids)}, data["feat"]


def label_index(label):
    """Return the class index for a five-class label."""
    return LABELS.index(label)


class PooledDataset(Dataset):
    """Pooled image and text features with a missing-image flag."""

    def __init__(
        self, manifest, image_cache=None, text_cache=None, image_key="b0",
        shuffle_text=False, seed=0,
    ):
        if image_cache is None:
            image_cache = CACHE / f"image_{image_key}.npz"
        if text_cache is None:
            text_cache = CACHE / "text_bert.npz"
        self.image_map, self.image_feat = load_index(
            image_cache, "image_id"
        )
        self.text_map, self.text_feat = load_index(
            text_cache, "tweet_id"
        )
        self.rows = manifest.reset_index(drop=True)
        self.image_dim = self.image_feat.shape[1]
        self.text_dim = self.text_feat.shape[1]
        self.text_perm = None
        if shuffle_text:
            rng = np.random.default_rng(seed)
            self.text_perm = rng.permutation(len(self.rows))
        self.image_mean = None
        self.image_std = None
        self.text_mean = None
        self.text_std = None

    def fit_stats(self):
        """Return train-set mean and standard deviation for both modalities."""
        image_sum = torch.zeros(self.image_dim)
        image_sq = torch.zeros(self.image_dim)
        text_sum = torch.zeros(self.text_dim)
        text_sq = torch.zeros(self.text_dim)
        n_image = 0
        n_text = 0
        for index in range(len(self.rows)):
            row = self.rows.iloc[index]
            if float(row["image_missing"]) == 0:
                image = torch.from_numpy(
                    self.image_feat[self.image_map[str(row["image_id"])]]
                ).float()
                image_sum += image
                image_sq += image ** 2
                n_image += 1
            text = torch.from_numpy(
                self.text_feat[self.text_map[str(row["tweet_id"])]]
            ).float()
            text_sum += text
            text_sq += text ** 2
            n_text += 1
        image_mean = image_sum / n_image
        image_std = (
            image_sq / n_image - image_mean ** 2
        ).clamp(min=1e-6).sqrt()
        text_mean = text_sum / n_text
        text_std = (text_sq / n_text - text_mean ** 2).clamp(min=1e-6).sqrt()
        return {
            "image_mean": image_mean,
            "image_std": image_std,
            "text_mean": text_mean,
            "text_std": text_std,
        }

    def apply_stats(self, stats):
        """Store train-set scaler statistics for this dataset."""
        self.image_mean = stats["image_mean"]
        self.image_std = stats["image_std"]
        self.text_mean = stats["text_mean"]
        self.text_std = stats["text_std"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = torch.zeros(self.image_dim)
        missing = float(row["image_missing"])
        if not missing:
            image = torch.from_numpy(
                self.image_feat[self.image_map[str(row["image_id"])]]
            ).float()
            if self.image_mean is not None:
                image = (image - self.image_mean) / self.image_std
        text_row = row
        if self.text_perm is not None:
            text_row = self.rows.iloc[self.text_perm[index]]
        text = torch.from_numpy(
            self.text_feat[self.text_map[str(text_row["tweet_id"])]]
        ).float()
        if self.text_mean is not None:
            text = (text - self.text_mean) / self.text_std
        label = label_index(row["label"])
        return image, text, torch.tensor(missing), torch.tensor(label)


class TokenDataset(Dataset):
    """EfficientNet-B0 and BERT token features for cross-attention."""

    def __init__(self, manifest, image_cache=None, text_cache=None):
        if image_cache is None:
            image_cache = CACHE / "image_tokens_b0.npz"
        if text_cache is None:
            text_cache = CACHE / "text_tokens_bert.npz"
        self.image_map, self.image_feat = load_index(
            image_cache, "image_id"
        )
        self.text_map, self.text_feat = load_index(
            text_cache, "tweet_id"
        )
        image_masks = np.load(image_cache, allow_pickle=True)["mask"]
        text_masks = np.load(text_cache, allow_pickle=True)["mask"]
        self.image_mask = image_masks
        self.text_mask = text_masks
        self.rows = manifest.reset_index(drop=True)
        self.image_dim = self.image_feat.shape[2]
        self.text_dim = self.text_feat.shape[2]
        self.image_mean = None
        self.image_std = None
        self.text_mean = None
        self.text_std = None

    def fit_stats(self):
        """Compute token statistics over valid tokens in these rows only."""
        image_sum = np.zeros(self.image_dim, dtype=np.float64)
        image_sq = np.zeros(self.image_dim, dtype=np.float64)
        text_sum = np.zeros(self.text_dim, dtype=np.float64)
        text_sq = np.zeros(self.text_dim, dtype=np.float64)
        image_n = 0
        text_n = 0
        for index in range(len(self.rows)):
            row = self.rows.iloc[index]
            image_i = self.image_map[str(row["image_id"])]
            image_mask = self.image_mask[image_i]
            if image_mask.any():
                image = self.image_feat[image_i][image_mask].astype(
                    np.float64
                )
                image_sum += image.sum(0)
                image_sq += (image ** 2).sum(0)
                image_n += image.shape[0]
            text_i = self.text_map[str(row["tweet_id"])]
            text_mask = self.text_mask[text_i]
            text = self.text_feat[text_i][text_mask].astype(np.float64)
            text_sum += text.sum(0)
            text_sq += (text ** 2).sum(0)
            text_n += text.shape[0]
        image_mean = image_sum / max(image_n, 1)
        text_mean = text_sum / max(text_n, 1)
        image_var = np.maximum(
            image_sq / max(image_n, 1) - image_mean**2, 1e-6
        )
        text_var = np.maximum(
            text_sq / max(text_n, 1) - text_mean**2, 1e-6
        )
        return {
            "image_mean": torch.from_numpy(image_mean).float(),
            "image_std": torch.from_numpy(np.sqrt(image_var)).float(),
            "text_mean": torch.from_numpy(text_mean).float(),
            "text_std": torch.from_numpy(np.sqrt(text_var)).float(),
        }

    def apply_stats(self, stats):
        """Store train-set scaler statistics for this dataset."""
        self.image_mean = stats["image_mean"]
        self.image_std = stats["image_std"]
        self.text_mean = stats["text_mean"]
        self.text_std = stats["text_std"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_i = self.image_map[str(row["image_id"])]
        text_i = self.text_map[str(row["tweet_id"])]
        image = torch.from_numpy(self.image_feat[image_i]).float()
        text = torch.from_numpy(self.text_feat[text_i]).float()
        image_mask = torch.from_numpy(self.image_mask[image_i]).bool()
        text_mask = torch.from_numpy(self.text_mask[text_i]).bool()
        if self.image_mean is not None:
            image = (image - self.image_mean) / self.image_std
            image[~image_mask] = 0.0
            text = (text - self.text_mean) / self.text_std
            text[~text_mask] = 0.0
        label = label_index(row["label"])
        return image, image_mask, text, text_mask, torch.tensor(label)


class TokenPooledDataset(TokenDataset):
    """Mean-pooled token features for the late-fusion control."""

    def fit_stats(self):
        """Compute pooled-feature statistics over valid train tokens."""
        image_sum = torch.zeros(self.image_dim)
        image_sq = torch.zeros(self.image_dim)
        text_sum = torch.zeros(self.text_dim)
        text_sq = torch.zeros(self.text_dim)
        n_image = 0
        n_text = 0
        for index in range(len(self.rows)):
            row = self.rows.iloc[index]
            image_i = self.image_map[str(row["image_id"])]
            text_i = self.text_map[str(row["tweet_id"])]
            image_mask = self.image_mask[image_i]
            text_mask = self.text_mask[text_i]
            if image_mask.any():
                image = self.image_feat[image_i][image_mask].astype(
                    np.float32
                ).mean(0)
                image_sum += torch.from_numpy(image)
                image_sq += torch.from_numpy(image ** 2)
                n_image += 1
            text = self.text_feat[text_i][text_mask].astype(
                np.float32
            ).mean(0)
            text_sum += torch.from_numpy(text)
            text_sq += torch.from_numpy(text ** 2)
            n_text += 1
        image_mean = image_sum / max(n_image, 1)
        text_mean = text_sum / max(n_text, 1)
        image_std = (
            image_sq / max(n_image, 1) - image_mean ** 2
        ).clamp(min=1e-6).sqrt()
        text_std = (
            text_sq / max(n_text, 1) - text_mean ** 2
        ).clamp(min=1e-6).sqrt()
        return {
            "image_mean": image_mean,
            "image_std": image_std,
            "text_mean": text_mean,
            "text_std": text_std,
        }

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_i = self.image_map[str(row["image_id"])]
        text_i = self.text_map[str(row["tweet_id"])]
        image_mask = self.image_mask[image_i]
        text_mask = self.text_mask[text_i]
        missing = 0.0 if image_mask.any() else 1.0
        image = torch.zeros(self.image_dim)
        if not missing:
            image = torch.from_numpy(
                self.image_feat[image_i][image_mask].astype(
                    np.float32
                ).mean(0)
            ).float()
            if self.image_mean is not None:
                image = (image - self.image_mean) / self.image_std
        text = torch.from_numpy(
            self.text_feat[text_i][text_mask].astype(np.float32).mean(0)
        ).float()
        if self.text_mean is not None:
            text = (text - self.text_mean) / self.text_std
        label = label_index(row["label"])
        return image, text, torch.tensor(missing), torch.tensor(label)
