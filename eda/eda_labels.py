"""Label and text exploration of the CrisisMMD v2.0 split files."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1])
ALL = ROOT / "crisismmd_datasplit_all"
AGREED = ROOT / "crisismmd_datasplit_agreed_label"
SPLITS = ["train", "dev", "test"]


def load(task, agreed=False):
    """Read the train, dev and test files of one task into one frame."""
    frames = []
    for split in SPLITS:
        if agreed:
            path = AGREED / f"task_{task}_text_img_agreed_lab_{split}.tsv"
        else:
            path = ALL / f"task_{task}_text_img_{split}.tsv"
        df = pd.read_csv(path, sep="\t", dtype=str, quoting=3)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def section(title):
    """Print a section header."""
    print(f"\n=== {title} ===")


tasks = {t: load(t) for t in ["informative", "humanitarian", "damage"]}
agreed = {t: load(t, True) for t in ["informative", "humanitarian"]}

section("Rows, unique tweets and images per split")
rows = []
for name, df in list(tasks.items()) + [
        (f"{t} agreed", d) for t, d in agreed.items()]:
    for split in SPLITS:
        s = df[df.split == split]
        rows.append([name, split, len(s), s.tweet_id.nunique(),
                     s.image_id.nunique()])
print(pd.DataFrame(rows, columns=["task", "split", "rows", "tweets",
                                  "images"]).to_string(index=False))

section("Images per tweet (informative, all splits)")
inf = tasks["informative"]
print(inf.groupby("tweet_id").size().value_counts().sort_index())

section("Tweets that appear in more than one split")
for name, df in list(tasks.items()) + list(agreed.items()):
    n = (df.groupby("tweet_id").split.nunique() > 1).sum()
    print(name, n)

section("Text label vs image label agreement (all data)")
for t in ["informative", "humanitarian"]:
    df = tasks[t]
    print(t, "agreement", round((df.label_text == df.label_image).mean(), 3))
    print(" label column equals text label",
          round((df.label == df.label_text).mean(), 3),
          "equals image label",
          round((df.label == df.label_image).mean(), 3))

section("Agreement by event (humanitarian)")
hum = tasks["humanitarian"]
by_event = hum.assign(agree=hum.label_text == hum.label_image).groupby(
    "event_name").agg(rows=("agree", "size"), agree=("agree", "mean"))
print(by_event.round(3).to_string())

section("Humanitarian: text label vs image label (counts)")
print(pd.crosstab(hum.label_text, hum.label_image, margins=True).to_string())

section("Class distribution per split")
for name, df, col in [("informative", tasks["informative"], "label"),
                      ("humanitarian", hum, "label"),
                      ("damage", tasks["damage"], "label"),
                      ("informative agreed", agreed["informative"], "label"),
                      ("humanitarian agreed", agreed["humanitarian"],
                       "label")]:
    print(f"\n{name}")
    print(pd.crosstab(df[col], df.split)[SPLITS].to_string())

section("Events per split (humanitarian agreed)")
ha = agreed["humanitarian"]
print(pd.crosstab(ha.event_name, ha.split)[SPLITS].to_string())

section("Class by event (humanitarian agreed, share of event rows)")
print(pd.crosstab(ha.event_name, ha.label, normalize="index")
      .round(2).to_string())

section("Text properties (unique tweets, informative all)")
tw = inf.drop_duplicates("tweet_id")
text = tw.tweet_text.fillna("")
words = text.str.split().str.len()
stats = {
    "tweets": len(tw),
    "words mean": round(words.mean(), 1),
    "words median": words.median(),
    "words max": words.max(),
    "retweets (RT @)": int(text.str.startswith("RT @").sum()),
    "with URL": int(text.str.contains("https?://").sum()),
    "with @mention": int(text.str.contains("@\\w").sum()),
    "with hashtag": int(text.str.contains("#\\w").sum()),
    "mojibake (â€ or Ã)": int(text.str.contains("â€|Ã").sum()),
    "non ASCII letters": int(text.apply(
        lambda s: bool(re.search("[^\\x00-\\x7F]", s))).sum()),
    "duplicate texts": int(text.duplicated().sum()),
}
for k, v in stats.items():
    print(k, v)

section("Duplicate texts after removing URLs and RT prefix")
clean = (text.str.replace(r"https?://\S+", "", regex=True)
         .str.replace(r"^RT @\w+:\s*", "", regex=True).str.strip()
         .str.lower())
dup = clean.duplicated(keep=False)
print("tweets sharing a cleaned text", int(dup.sum()))
groups = tw[dup].assign(c=clean[dup]).groupby("c")
cross = groups.split.nunique()
print("cleaned texts spread over more than one split",
      int((cross > 1).sum()))

section("Damage task overlap with humanitarian images")
print("damage images also in humanitarian",
      tasks["damage"].image_id.isin(hum.image_id).sum(),
      "of", len(tasks["damage"]))
