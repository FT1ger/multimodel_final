"""Build local manifests for the CrisisMMD agreed splits."""
import argparse
from pathlib import Path

import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
PROJECT = FINAL.parent
DEFAULT_DATA = PROJECT / "data" / "CrisisMMD"
DEFAULT_SPLITS = FINAL / "data" / "splits"
DEFAULT_AUDIT = FINAL / "eda" / "image_audit.csv"
DEFAULT_PLACEHOLDERS = FINAL / "assets" / "placeholder_ids.csv"
DEFAULT_OUT = FINAL / "data"

LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian",
]

MAP_8_TO_5 = {
    "affected_individuals": "affected_individuals",
    "injured_or_dead_people": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "infrastructure_and_utility_damage": "infrastructure_and_utility_damage",
    "vehicle_damage": "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort": (
        "rescue_volunteering_or_donation_effort"
    ),
    "other_relevant_information": "other_relevant_information",
    "not_humanitarian": "not_humanitarian",
}


def read_tsv(path):
    """Read a split TSV, keeping tweet text exactly as stored."""
    return pd.read_csv(path, sep="\t", dtype=str, quoting=3)


def map_label(value):
    """Map an original eight-class label to the five-class label."""
    return MAP_8_TO_5.get(str(value), None)


def detect_placeholders(audit_path):
    """Return placeholder image ids from the image audit table."""
    audit = pd.read_csv(
        audit_path, dtype={"ahash": str, "image_id": str}
    )
    ifttt = (
        audit["ahash"]
        == "0011111100111111111111111000000110000001111111111111111111111111"
    )
    email = (
        audit["ahash"]
        == "1111111111111111100100010000000100000001111111111111111111111111"
    )
    not_found = (
        audit["ahash"]
        == "0000000001111110011111100110011001000010011111100111111000000000"
    )
    rows = []
    for mask, name in [
        (ifttt, "ifttt_image_not_found"),
        (email, "email_this"),
        (not_found, "image_not_found"),
    ]:
        sub = audit.loc[mask, ["image_id", "width", "height"]].copy()
        sub["placeholder_type"] = name
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates("image_id")
    return out


def load_placeholders(placeholder_path, audit_path):
    """Load committed placeholder ids, or detect them from the audit."""
    if Path(placeholder_path).exists():
        return pd.read_csv(placeholder_path, dtype={"image_id": str})
    return detect_placeholders(audit_path)


def add_columns(df, split, data_root, placeholders):
    """Add split, paths, five-class labels and the missing-image flag."""
    df = df.copy()
    df["split"] = split
    df["label"] = df["label"].map(map_label)
    df["label_text_5"] = df["label_text"].map(map_label)
    df["label_image_5"] = df["label_image"].map(map_label)
    df["image_missing"] = df["image_id"].isin(placeholders).astype(int)
    df["image_path"] = df["image"].map(lambda p: str(data_root / p))
    df["path_exists"] = df["image_path"].map(lambda p: Path(p).exists())
    return df


def build_manifest(splits_dir, data_root, audit_path, placeholder_path):
    """Build the agreed-split and all-split manifests."""
    placeholders = set(
        load_placeholders(placeholder_path, audit_path)["image_id"]
    )

    agreed_dir = splits_dir / "crisismmd_datasplit_agreed_label"
    frames = []
    for split in ["train", "dev", "test"]:
        path = agreed_dir / (
            f"task_humanitarian_text_img_agreed_lab_{split}.tsv"
        )
        df = read_tsv(path)
        frames.append(add_columns(df, split, data_root, placeholders))
    agreed = pd.concat(frames, ignore_index=True)

    all_dir = splits_dir / "crisismmd_datasplit_all"
    frames = []
    for split in ["train", "dev", "test"]:
        path = all_dir / f"task_humanitarian_text_img_{split}.tsv"
        df = read_tsv(path)
        frames.append(add_columns(df, split, data_root, placeholders))
    all_rows = pd.concat(frames, ignore_index=True)
    return agreed, all_rows, placeholders


def main():
    """Write the manifests and print aggregate checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--placeholders", type=Path, default=DEFAULT_PLACEHOLDERS
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    agreed, all_rows, placeholders = build_manifest(
        args.splits_dir, args.data_root, args.audit, args.placeholders
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    agreed.to_csv(args.out_dir / "manifest_agreed.csv", index=False)
    all_rows.to_csv(args.out_dir / "manifest_all.csv", index=False)
    load_placeholders(
        args.placeholders, args.audit
    ).to_csv(args.out_dir / "placeholder_ids.csv", index=False)

    print("agreed rows by split")
    print(agreed["split"].value_counts().to_string())
    print("\nagreed labels by split")
    print(pd.crosstab(agreed["label"], agreed["split"]).to_string())
    print("\nmissing images by split")
    print(
        agreed.groupby("split")["image_missing"].sum().to_string()
    )
    print("\nunresolved labels")
    print(agreed["label"].isna().sum(), "agreed")
    print(all_rows["label_text_5"].isna().sum(), "all text")
    print(all_rows["label_image_5"].isna().sum(), "all image")
    print("\nmissing paths")
    print(int((~agreed["path_exists"]).sum()))
    print("\nplaceholders")
    counts = load_placeholders(
        args.placeholders, args.audit
    )["placeholder_type"]
    print(counts.value_counts().to_string())


if __name__ == "__main__":
    main()
