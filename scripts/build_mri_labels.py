"""Build the MRI label file for training from the IDA export, for everyone converted so far.

train_mri.py and mri_bundle.py only use people who are in the label file. The original file
(data/interim/pp1_mri_subset.csv) lists the first 60 people, so people downloaded later are
silently left out of training until they are added. This script writes a label file for
every person in a conversion table (data/interim/*conversion*.csv):

* group, sex and age come from the IDA export row of the image that was actually converted,
  so the age is the age at that scan's visit;
* people kept for testing the app are left out, so they can never enter training;
* a person whose group is not PD or Control, or who cannot be matched, is reported and left
  out, never guessed.

    python scripts/build_mri_labels.py                      # writes data/interim/mri_labels.csv
    python scripts/build_mri_labels.py --held-out 3102 3104 3122 177956

Then train with:  python models/mri/src/train_mri.py --labels data/interim/mri_labels.csv
Every count printed comes from the files. Nothing here is committed: data/ is git-ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEARCH_GLOB = "data/raw/ida/idaSearch_*.csv"
# Kept out of training on purpose: used to test the desktop app end to end.
#   3102, 3104: picked before PP1 (3104 is the Control test case)
#   3122 (TEST-A), 177956 (TEST-B): PD test cases from the 20 Oct download
DEFAULT_HELD_OUT = ("3102", "3104", "3122", "177956")
GROUPS = {"PD", "Control"}


def converted_images() -> pd.DataFrame:
    """subject_id, image_id of every successfully converted series."""
    frames = []
    for table in sorted((ROOT / "data/interim").glob("*conversion*.csv")):
        df = pd.read_csv(table, dtype=str)
        if {"subject_id", "image_id"} <= set(df.columns):
            ok = df[df.get("status", "ok").fillna("") == "ok"] if "status" in df else df
            frames.append(ok[["subject_id", "image_id"]])
    if not frames:
        raise SystemExit("no conversion tables in data/interim")
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data/interim/mri_labels.csv")
    ap.add_argument("--held-out", nargs="*", default=list(DEFAULT_HELD_OUT))
    args = ap.parse_args()

    search = pd.read_csv(sorted(ROOT.glob(SEARCH_GLOB))[-1], dtype=str)
    images = converted_images()
    rows = images.merge(
        search, left_on=["subject_id", "image_id"], right_on=["Subject ID", "Image ID"], how="left"
    )
    unmatched = sorted(set(rows.loc[rows["Subject ID"].isna(), "subject_id"]))
    rows = rows.dropna(subset=["Subject ID"])

    # One row per person: the visit of their converted images (the earliest, if several).
    rows["_date"] = pd.to_datetime(rows["Study Date"], format="%m/%d/%Y", errors="coerce")
    person = rows.sort_values("_date").groupby("Subject ID", as_index=False).first()
    held = person["Subject ID"].isin(set(args.held_out))
    other_group = ~person["Research Group"].isin(GROUPS)
    keep = person[~held & ~other_group]

    out = keep[["Subject ID", "Visit", "Research Group", "Sex", "Age", "Study Date"]]
    args.out.write_text(out.to_csv(index=False))

    print(f"people with a converted series: {images['subject_id'].nunique()}")
    print(f"  not found in the IDA export (left out): {len(unmatched)}")
    print(f"  held out for app testing: {int(held.sum())}")
    print(f"  other research group (left out): {int(other_group.sum())}")
    counts = out["Research Group"].value_counts().to_dict()
    print(f"label file: {len(out)} people {counts} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
