"""Pick the next PPMI subjects to download for the MRI model: T1 + DTI from the same visit.

Reads the IDA search export (data/raw/ida/idaSearch_*.csv), drops every subject already
downloaded or processed (or already listed in an earlier batch), keeps subjects that have a
real 3D T1 and a DTI series on the SAME visit date, and picks a batch that is balanced between
PD and Control and spread across DTI acquisition families (a stand-in for scanner variety,
because the search export has no scanner field). It writes the image IDs to paste into IDA's
"download by image ID" box, and a table saying who was picked and why.

    python scripts/next_mri_download_batch.py --n 10 --out data/interim/dti_next5_image_ids.txt

Every count printed here comes from the files, not from memory. Nothing it reads or writes
is committed: data/ is git-ignored (CLAUDE.md section 4).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEARCH_GLOB = "data/raw/ida/idaSearch_*.csv"

# A 3D T1 that the T1 pipeline can use. "2D GRE-MT", "2D GRE-NM", rs-fMRI and localisers are
# in the same export, so the description decides, not the weighting.
T1_YES = re.compile(
    r"mprage|3d[\s_-]*t1|t1[\s_-]*3d|fspgr|spgr|bravo|t1[\s_-]*weighted|sag.*t1|t1.*sag", re.I
)
T1_NO = re.compile(r"gre|mt\b|nm\b|loc|fmri|bold|flair|t2|pd\b|swi|asl|survey|cal", re.I)

DTI_YES = re.compile(r"dti|dwi|diff", re.I)
# Maps the scanner computed from the diffusion data. They are not the diffusion series, so
# they are never downloaded (our pipeline makes its own FA and MD).
DTI_DERIVED = re.compile(r"adc|(?:^|[_\s])fa(?:$|[_\s])|trace|colfa|color|tensor|_map|exp", re.I)

# Subjects kept out of training on purpose: the two used to test the desktop app end to end.
DEFAULT_HELD_OUT = ("3102", "3104")


def downloaded_subjects() -> set[str]:
    """Subjects already on disk or already converted/processed."""
    seen: set[str] = set()
    for folder in ("data/raw/PPMI", "data/raw/PPMI 2", "data/raw/PPMI 3", "data/processed/nifti"):
        path = ROOT / folder
        if path.is_dir():
            seen.update(p.name for p in path.iterdir() if p.is_dir())
    for table in (ROOT / "data/interim").glob("*conversion*.csv"):
        frame = pd.read_csv(table, usecols=["subject_id"])
        seen.update(str(s) for s in frame["subject_id"].unique())
    return seen


def already_listed(out: Path) -> set[str]:
    """Subjects in earlier batches this script wrote, so a batch you have not downloaded yet
    is not picked again. The batch being written now is left out of the check."""
    seen: set[str] = set()
    for table in (ROOT / "data/interim").glob("dti_next*_image_ids.csv"):
        if table.resolve() != out.with_suffix(".csv").resolve():
            seen.update(str(s) for s in pd.read_csv(table, usecols=["subject_id"])["subject_id"])
    return seen


def dti_family(description: str) -> str:
    """A coarse acquisition family from the series name, so a batch is not all one protocol."""
    d = description.lower()
    if "gated" in d:
        return "gated"
    if "64dir" in d or "b2000" in d or "b700" in d or "b1000" in d:
        return "multishell_pa"
    if re.search(r"(^|_)(lr|rl)(_|$)", d) or "l-r" in d or "r-l" in d:
        return "lr_rl"
    if "ep2d" in d:
        return "ep2d"
    return "other"


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per subject-visit that has a usable 3D T1 and at least one DTI series."""
    frame = frame.copy()
    frame["Subject ID"] = frame["Subject ID"].astype(str)
    desc = frame["Description"].fillna("")
    is_t1 = desc.str.contains(T1_YES) & ~desc.str.contains(T1_NO) & (frame["Modality"] == "MRI")
    is_dti = (
        desc.str.contains(DTI_YES)
        & ~desc.str.contains(r"fmri|bold", case=False)
        & ~desc.str.contains(DTI_DERIVED)
    )
    keys = ["Subject ID", "Visit", "Study Date"]
    both = frame[is_t1][keys].drop_duplicates().merge(frame[is_dti][keys].drop_duplicates(), on=keys)
    rows = []
    for _, key in both.iterrows():
        m = (
            (frame["Subject ID"] == key["Subject ID"])
            & (frame["Visit"] == key["Visit"])
            & (frame["Study Date"] == key["Study Date"])
        )
        t1_ids = frame[m & is_t1]
        dti_ids = frame[m & is_dti]
        first = frame[m].iloc[0]
        rows.append(
            {
                "subject_id": key["Subject ID"],
                "group": first["Research Group"],
                "sex": first["Sex"],
                "age": first["Age"],
                "visit": key["Visit"],
                "study_date": key["Study Date"],
                "t1_ids": list(t1_ids["Image ID"].astype(str)),
                "t1_desc": t1_ids["Description"].iloc[0],
                "dti_ids": list(dti_ids["Image ID"].astype(str)),
                "dti_desc": ", ".join(sorted(set(dti_ids["Description"]))),
                "dti_family": dti_family(dti_ids["Description"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def pick(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    """Baseline visit first, one visit per subject, then balance PD/Control and DTI families."""
    candidates = candidates.assign(order=candidates["visit"].map(lambda v: 0 if v == "Baseline" else 1))
    candidates = candidates.sort_values(["order", "subject_id"]).drop_duplicates("subject_id")
    chosen = []
    for group in ("PD", "Control"):
        pool = candidates[candidates["group"] == group]
        # Round-robin over DTI families so the batch is not one protocol.
        buckets = {f: list(p.itertuples(index=False)) for f, p in pool.groupby("dti_family")}
        taken = 0
        while taken < n // 2 and any(buckets.values()):
            for family in sorted(buckets):
                if buckets[family] and taken < n // 2:
                    chosen.append(buckets[family].pop(0))
                    taken += 1
    return pd.DataFrame(chosen).drop(columns=["order"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=10, help="subjects in the batch (half PD, half Control)")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_HELD_OUT),
        help="subject IDs to keep out (default: the two app-test subjects)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/interim/dti_next5_image_ids.txt")
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="also write the batch as small files of this many subjects each (for slow links)",
    )
    args = parser.parse_args()
    args.out = args.out if args.out.is_absolute() else ROOT / args.out

    search = sorted(ROOT.glob(SEARCH_GLOB))[-1]
    frame = pd.read_csv(search)
    have = downloaded_subjects() | set(args.exclude) | already_listed(args.out)
    everything = eligible(frame)
    candidates = everything[~everything["subject_id"].isin(have)]

    print(f"search file: {search.name}")
    print(f"subjects with a 3D T1 and DTI on the same visit: {everything.subject_id.nunique()}")
    print(f"  downloaded, processed, held out or already listed: {everything[everything.subject_id.isin(have)].subject_id.nunique()}")
    print(f"  still available: {candidates.subject_id.nunique()}  {candidates.drop_duplicates('subject_id').group.value_counts().to_dict()}")

    batch = pick(candidates, args.n)
    ids = [i for _, row in batch.iterrows() for i in (*row["t1_ids"], *row["dti_ids"])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(",".join(ids) + "\n")
    table = args.out.with_suffix(".csv")
    batch.assign(t1_ids=batch.t1_ids.map(",".join), dti_ids=batch.dti_ids.map(",".join)).to_csv(
        table, index=False
    )

    if args.chunk > 0:
        for start in range(0, len(batch), args.chunk):
            part = batch.iloc[start : start + args.chunk]
            part_ids = [i for _, row in part.iterrows() for i in (*row["t1_ids"], *row["dti_ids"])]
            name = args.out.with_name(f"{args.out.stem}_part{start // args.chunk + 1}.txt")
            name.write_text(",".join(part_ids) + "\n")
            print(f"  {name.name}: subjects {', '.join(part.subject_id)} ({len(part_ids)} IDs)")

    print(f"\nbatch: {len(batch)} subjects {batch.group.value_counts().to_dict()}, {len(ids)} image IDs")
    print(f"DTI families: {batch.dti_family.value_counts().to_dict()}")
    print(f"wrote {args.out.relative_to(ROOT)} and {table.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
