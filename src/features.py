"""Extract pooled and token-level features for CrisisMMD."""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from preprocess import clean_text, eval_transform  # noqa: E402

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_AGREED = FINAL / "data" / "manifest_agreed.csv"
DEFAULT_ALL = FINAL / "data" / "manifest_all.csv"
DEFAULT_CACHE = FINAL / "cache"


class ImageDataset(Dataset):
    """Load images from manifest rows with a missing-image flag."""

    def __init__(self, paths, missing, transform):
        self.paths = list(paths)
        self.missing = [int(float(x)) == 1 for x in missing]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        missing = bool(self.missing[index])
        if missing:
            return torch.zeros(3, 224, 224), 1.0
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, 0.0
        except (OSError, ValueError):
            return torch.zeros(3, 224, 224), 1.0


def load_pooled(path):
    """Load a pooled feature cache keyed by image_id."""
    data = np.load(path, allow_pickle=True)
    return data["image_id"], data["feat"].astype(np.float32)


def unique_images(manifest):
    """Return one row per image in a manifest."""
    cols = ["image_id", "image_path", "image_missing"]
    return manifest[cols].drop_duplicates("image_id").reset_index(drop=True)


def extract_pooled_b0(manifest, out_path, device):
    """Extract frozen EfficientNet-B0 pooled features for all images."""
    rows = unique_images(manifest)
    dataset = ImageDataset(
        rows["image_path"], rows["image_missing"], eval_transform()
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier = torch.nn.Identity()
    model = model.eval().to(device)
    out = []
    start = time.time()
    with torch.no_grad():
        for images, _ in loader:
            out.append(model(images.to(device)).float().cpu().numpy())
    features = np.concatenate(out, axis=0)
    np.savez(
        out_path,
        image_id=rows["image_id"].values,
        feat=features.astype(np.float16),
    )
    print(f"pooled images {len(features)} in {time.time() - start:.0f} s")


def extract_pooled_bert(manifest, out_path, device):
    """Extract mean-pooled BERT features over real tokens."""
    tweets = manifest.drop_duplicates("tweet_id").reset_index(drop=True)
    texts = [clean_text(x) for x in tweets["tweet_text"].fillna("")]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval().to(device)
    out = []
    start = time.time()
    with torch.no_grad():
        for start_i in range(0, len(texts), 64):
            batch = texts[start_i:start_i + 64]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            ).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            out.append(pooled.float().cpu().numpy().astype(np.float16))
    np.savez(
        out_path,
        tweet_id=tweets["tweet_id"].values,
        feat=np.concatenate(out, axis=0),
    )
    print(f"pooled tweets {len(tweets)} in {time.time() - start:.0f} s")


def extract_pooled_r50(manifest, out_path, device):
    """Extract frozen ResNet-50 pooled features for all images."""
    rows = unique_images(manifest)
    dataset = ImageDataset(
        rows["image_path"], rows["image_missing"], eval_transform()
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    )
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)
    out = []
    start = time.time()
    with torch.no_grad():
        for images, _ in loader:
            out.append(model(images.to(device)).float().cpu().numpy())
    features = np.concatenate(out, axis=0)
    np.savez(
        out_path,
        image_id=rows["image_id"].values,
        feat=features.astype(np.float16),
    )
    print(f"images {len(features)} in {time.time() - start:.0f} s")


def extract_tokens_b0(manifest, out_path, device):
    """Extract EfficientNet-B0 7x7 token features for agreed images."""
    rows = unique_images(manifest)
    dataset = ImageDataset(
        rows["image_path"], rows["image_missing"], eval_transform()
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model = model.eval().to(device)
    feats = []
    masks = []
    start = time.time()
    with torch.no_grad():
        for images, missing in loader:
            maps = model.features(images.to(device))
            tokens = maps.flatten(2).transpose(1, 2)
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool)
            mask[missing.bool()] = False
            feats.append(tokens.float().cpu().numpy().astype(np.float16))
            masks.append(mask.numpy())
    np.savez(
        out_path,
        image_id=rows["image_id"].values,
        feat=np.concatenate(feats, axis=0),
        mask=np.concatenate(masks, axis=0),
    )
    print(f"image tokens {len(rows)} in {time.time() - start:.0f} s")


def extract_tokens_bert(manifest, out_path, device):
    """Extract BERT last-layer token features for agreed tweets."""
    tweets = manifest.drop_duplicates("tweet_id").reset_index(drop=True)
    texts = [clean_text(x) for x in tweets["tweet_text"].fillna("")]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval().to(device)
    feats = []
    masks = []
    start = time.time()
    with torch.no_grad():
        for start_i in range(0, len(texts), 32):
            batch = texts[start_i:start_i + 32]
            enc = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            hidden = model(**enc).last_hidden_state
            feats.append(hidden.float().cpu().numpy().astype(np.float16))
            masks.append(enc["attention_mask"].bool().cpu().numpy())
    np.savez(
        out_path,
        tweet_id=tweets["tweet_id"].values,
        feat=np.concatenate(feats, axis=0),
        mask=np.concatenate(masks, axis=0),
    )
    print(f"text tokens {len(tweets)} in {time.time() - start:.0f} s")


def main():
    """Run the requested feature extraction steps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreed", type=Path, default=DEFAULT_AGREED)
    parser.add_argument("--all", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--what",
        choices=["pooled", "r50", "tokens", "text", "all"],
        default="all",
    )
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("device", device)
    agreed = pd.read_csv(args.agreed, dtype=str)
    all_rows = pd.read_csv(args.all, dtype=str)
    if args.what in {"pooled", "all"}:
        extract_pooled_b0(
            all_rows, args.cache / "image_b0.npz", device
        )
        extract_pooled_bert(
            all_rows, args.cache / "text_bert.npz", device
        )
    if args.what in {"r50", "all"}:
        extract_pooled_r50(
            all_rows, args.cache / "image_r50.npz", device
        )
    if args.what in {"tokens", "all"}:
        extract_tokens_b0(
            agreed, args.cache / "image_tokens_b0.npz", device
        )
    if args.what in {"text", "all"}:
        extract_tokens_bert(
            agreed, args.cache / "text_tokens_bert.npz", device
        )


if __name__ == "__main__":
    main()
