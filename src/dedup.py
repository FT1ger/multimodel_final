"""Build pHash and dHash duplicate flags for the agreed test split."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from preprocess import clean_text

FINAL = Path(__file__).resolve().parents[1]
DATA = FINAL.parent / "data" / "CrisisMMD"
AUDIT = FINAL / "eda" / "image_audit.csv"
PHASH = FINAL / "cache" / "near_duplicates_humanitarian.csv"
OUT = FINAL / "results" / "report"


def dhash(path, size=16):
    """Return a 256-bit difference hash for one image."""
    with Image.open(path) as image:
        gray = np.asarray(
            image.convert("L").resize((size + 1, size), Image.BILINEAR),
            dtype=float,
        )
    return "".join(
        "1" if bit else "0" for bit in (gray[:, 1:] > gray[:, :-1]).ravel()
    )


def main():
    """Write pHash, dHash and union flags for agreed test images."""
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(
        FINAL / "data" / "manifest_agreed.csv", dtype=str
    )
    test_ids = set(manifest.loc[manifest.split == "test", "image_id"])

    pairs = pd.read_csv(PHASH)
    phash_ids = set(pairs.loc[pairs.split_A == "test", "image_id_A"])
    phash_ids |= set(pairs.loc[pairs.split_B == "test", "image_id_B"])

    audit = pd.read_csv(
        AUDIT, dtype={"ahash": str, "image_id": str, "image": str}
    )
    near = audit[audit.ahash.duplicated(keep=False)].copy()
    near["dhash"] = [
        dhash(DATA / path) for path in near["image"]
    ]
    dhash_ids = set()
    for _, group in near.groupby("dhash"):
        splits = set(group["split"])
        if "test" in splits and ({"train", "dev"} & splits):
            dhash_ids |= set(
                group.loc[group.split == "test", "image_id"]
            )

    dhash_agreed = dhash_ids & test_ids
    union = (phash_ids | dhash_agreed) & test_ids

    manifest["clean"] = manifest["tweet_text"].map(clean_text)
    train_text = set(
        manifest.loc[manifest.split == "train", "clean"]
    )
    dev_text = set(manifest.loc[manifest.split == "dev", "clean"])
    text_ids = set(
        manifest.loc[
            (manifest.split == "test")
            & (
                manifest["clean"].isin(train_text)
                | manifest["clean"].isin(dev_text)
            ),
            "image_id",
        ]
    )

    rows = []
    for image_id in sorted(test_ids):
        rows.append({
            "image_id": image_id,
            "phash": int(image_id in phash_ids),
            "dhash": int(image_id in dhash_agreed),
            "union": int(image_id in union),
            "text": int(image_id in text_ids),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "dedup_flags.csv", index=False)
    print("pHash agreed test", len(phash_ids & test_ids))
    print("dHash agreed test", len(dhash_agreed))
    print("dHash new vs pHash", len(dhash_agreed - phash_ids))
    print("union agreed test", len(union))
    print("text-cross-split agreed test", len(text_ids))


if __name__ == "__main__":
    main()
