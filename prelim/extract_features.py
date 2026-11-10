"""Extract frozen EfficientNet-B0 image and BERT text features for CrisisMMD.

Usage: python extract_features.py DATA_DIR SPLITS_DIR OUT_DIR
DATA_DIR holds data_image/, SPLITS_DIR holds crisismmd_datasplit_all/.
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

TF = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])


class Images(Dataset):
    """Load and transform images from a list of paths."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return TF(im.convert("RGB"))


def fix_mojibake(text):
    """Undo UTF-8 text that was decoded as cp1252, when that is the case."""
    if "â" not in text and "Ã" not in text:
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clean_text(text):
    """Fix encoding, drop the RT prefix, mask URLs and user mentions."""
    text = fix_mojibake(text)
    text = re.sub(r"^RT @\w+:\s*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


def load_index(splits_dir):
    """All image rows of the humanitarian files (same rows as informative)."""
    frames = [pd.read_csv(splits_dir / f"task_humanitarian_text_img_{s}.tsv",
                          sep="\t", dtype=str, quoting=3)
              for s in ["train", "dev", "test"]]
    return pd.concat(frames, ignore_index=True)


def image_features(paths, device):
    """Pooled 1280-d EfficientNet-B0 features, one row per image."""
    model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier = torch.nn.Identity()
    model = model.eval().to(device)
    out = []
    with torch.no_grad():
        for batch in DataLoader(Images(paths), batch_size=64):
            out.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def text_features(texts, device):
    """Mean-pooled 768-d BERT features over real tokens."""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], padding=True, truncation=True,
                      max_length=96, return_tensors="pt").to(device)
            hidden = bert(**enc).last_hidden_state
            mask = enc.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            out.append(pooled.cpu().numpy())
    return np.concatenate(out)


def main():
    """Extract and save both feature sets."""
    data_dir, splits_dir, out_dir = map(Path, sys.argv[1:4])
    splits_dir = splits_dir / "crisismmd_datasplit_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    index = load_index(splits_dir)

    t = time.time()
    img = image_features([data_dir / p for p in index.image], device)
    np.savez(out_dir / "image_b0.npz", image_id=index.image_id.values,
             feat=img.astype(np.float16))
    print(f"images {len(img)} in {time.time() - t:.0f} s")

    tweets = index.drop_duplicates("tweet_id")
    texts = [clean_text(x) for x in tweets.tweet_text.fillna("")]
    t = time.time()
    txt = text_features(texts, device)
    np.savez(out_dir / "text_bert.npz", tweet_id=tweets.tweet_id.values,
             feat=txt.astype(np.float16))
    print(f"tweets {len(txt)} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
