"""Check image duplicates in CrisisMMD: exact (MD5) and near (aHash)."""
import hashlib
import sys
from pathlib import Path

import pandas as pd

DATA = Path(sys.argv[1])
audit = pd.read_csv(Path(sys.argv[2]) / "image_audit.csv", dtype={
    "ahash": str, "tweet_id": str})


def md5(path):
    """MD5 of a file."""
    return hashlib.md5(path.read_bytes()).hexdigest()


audit["md5"] = [md5(DATA / p) for p in audit.image]


def report(key, name):
    """Print duplicate group counts for one hash column."""
    dup = audit[audit[key].duplicated(keep=False)]
    g = dup.groupby(key)
    other_tweet = g.tweet_id.nunique() > 1
    print(f"\n{name}: images {len(dup)}, groups {dup[key].nunique()}")
    print(" groups from more than one tweet", int(other_tweet.sum()))
    print(" groups across splits", int((g.split.nunique() > 1).sum()))
    print(" groups across events", int((g.event_name.nunique() > 1).sum()))
    print(" groups with different image labels",
          int((g.label_image.nunique() > 1).sum()))
    return dup


report("md5", "Exact duplicates (MD5)")
near = report("ahash", "Near duplicates (aHash)")
flat = near.ahash.isin(["0" * 64, "1" * 64]) | (near.contrast < 5)
print(" of which flat images (uniform hash or contrast under 5)",
      int(flat.sum()))
sizes = near.groupby("ahash").size().sort_values(ascending=False)
print(" largest aHash groups", sizes.head(5).tolist())

# images that pass the near check and also come from different tweets
# in different splits are the leakage candidates
g = near.groupby("ahash")
leak = g.filter(lambda x: x.split.nunique() > 1 and x.tweet_id.nunique() > 1)
print("\nleakage candidates (aHash, different tweets, across splits):",
      len(leak), "images in", leak.ahash.nunique(), "groups")
test_leak = leak[leak.split == "test"]
print(" test images with a near copy in train or dev", len(test_leak))


def dhash(path, size=16):
    """256-bit difference hash, finer than the 8x8 average hash."""
    from PIL import Image
    import numpy as np
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size),
                                              Image.BILINEAR), dtype=float)
    return "".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).ravel())


# recheck the aHash candidates with the finer hash
near = near.copy()
near["dhash"] = [dhash(DATA / p) for p in near.image]
fine = near[near.dhash.duplicated(keep=False)]
gf = fine.groupby("dhash")
print("\nNear duplicates confirmed by 16x16 dHash:", len(fine), "images in",
      fine.dhash.nunique(), "groups")
print(" groups across splits", int((gf.split.nunique() > 1).sum()))
print(" groups with different image labels",
      int((gf.label_image.nunique() > 1).sum()))
cross = gf.filter(lambda x: x.split.nunique() > 1)
print(" test images with a confirmed copy in train or dev",
      int((cross.split == "test").sum()))
big = gf.size().sort_values(ascending=False).head(5)
for h, n in big.items():
    print(" group of", n, fine[fine.dhash == h].image.iloc[0])
