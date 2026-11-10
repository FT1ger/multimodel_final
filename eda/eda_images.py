"""Image exploration of CrisisMMD v2.0: integrity, size, sharpness, duplicates."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA = Path(sys.argv[1])  # folder that holds data_image/
SPLITS_DIR = Path(sys.argv[2]) / "crisismmd_datasplit_all"
OUT = Path(sys.argv[3])


def load_index():
    """Read every image row of the humanitarian split files."""
    frames = []
    for split in ["train", "dev", "test"]:
        path = SPLITS_DIR / f"task_humanitarian_text_img_{split}.tsv"
        df = pd.read_csv(path, sep="\t", dtype=str, quoting=3)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def laplacian_var(gray):
    """Variance of the 4-neighbour Laplacian, a standard sharpness score."""
    g = gray.astype(np.float32)
    lap = (g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1]
           - 4 * g[1:-1, 1:-1])
    return float(lap.var())


def average_hash(gray_img):
    """64-bit average hash of an image, used to find near duplicates."""
    small = np.asarray(gray_img.resize((8, 8), Image.BILINEAR),
                       dtype=np.float32)
    bits = (small > small.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)


def describe(path):
    """Open one image and return its basic properties."""
    try:
        with Image.open(path) as im:
            fmt, mode, (w, h) = im.format, im.mode, im.size
            im.load()
            gray = im.convert("L")
    except Exception as err:  # corrupt or missing file
        return {"ok": False, "error": type(err).__name__}
    small = gray.copy()
    small.thumbnail((512, 512))
    arr = np.asarray(small)
    return {"ok": True, "format": fmt, "mode": mode, "width": w,
            "height": h, "bytes": path.stat().st_size,
            "sharpness": laplacian_var(arr),
            "brightness": float(arr.mean()),
            "contrast": float(arr.std()),
            "ahash": average_hash(gray)}


index = load_index()
props = pd.DataFrame([describe(DATA / p) for p in index.image])
df = pd.concat([index, props], axis=1)
df.drop(columns=["tweet_text"]).to_csv(OUT / "image_audit.csv", index=False)

print("images", len(df), "readable", int(df.ok.sum()))
if (~df.ok).any():
    print(df.loc[~df.ok, "error"].value_counts().to_string())
ok = df[df.ok]
print("\nformat\n" + ok.format.value_counts().to_string())
print("\nmode\n" + ok["mode"].value_counts().to_string())
print("\nsize summary")
print(ok[["width", "height", "bytes"]].describe().round(0).to_string())
print("\nshorter side under 224:", int((ok[["width", "height"]].min(1)
                                       < 224).sum()))
print("\nsharpness, brightness, contrast")
print(ok[["sharpness", "brightness", "contrast"]].describe().round(1)
      .to_string())
blur_cut = 100
print(f"\nsharpness under {blur_cut} (likely blurry):",
      int((ok.sharpness < blur_cut).sum()))
print("\nsharpness median by class")
print(ok.groupby("label_image").sharpness.median().round(0).to_string())

dup = ok[ok.ahash.duplicated(keep=False)]
print("\nimages sharing an average hash:", len(dup),
      "groups:", dup.ahash.nunique())
spread = dup.groupby("ahash").split.nunique()
print("hash groups spread over more than one split:",
      int((spread > 1).sum()))
label_mix = dup.groupby("ahash").label_image.nunique()
print("hash groups with more than one image label:",
      int((label_mix > 1).sum()))
