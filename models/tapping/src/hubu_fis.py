"""Load and check the HUBU-FIS finger-tapping labels and videos (task FT-06).

The label CSV (``fis_diagnostic.csv``, the dataset's own file name) has one row
per clip and two columns, ``ID`` and ``UPDRS``. The UPDRS value is a
clinician's MDS-UPDRS 3.4 finger-tapping rating for that hand. It is a label
or a stratification variable, never a model input (CLAUDE.md section 3.1).

Every check here raises ``HubuFisError`` and never warns:

- the CSV has exactly the columns ``ID`` and ``UPDRS``;
- every ``ID`` parses (``clip_id.parse_clip_id``) and appears once;
- every ``UPDRS`` is one of 0-4 written as a plain integer. A blank is an error,
  never read as 0;
- no participant number is spelled two ways (``CONTROL1`` and ``CONTROL01``),
  which would split one person into two participant keys;
- no two video files share a stem (``X.mp4`` and ``X.MOV``);
- every CSV row has a video and every video has a CSV row.

Errors point at CSV line numbers or hashed IDs (``pseudonym``), never raw IDs.

Video suffixes are compared lower-cased, so ``.MOV`` and ``.mp4`` match the
same way on macOS (case-insensitive file system) and Linux (case-sensitive).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from clip_id import UnparseableClipId, parse_clip_id

LABEL_COLUMNS = ("ID", "UPDRS")
UPDRS_VALUES = (0, 1, 2, 3, 4)
VIDEO_SUFFIXES = (".mp4", ".mov")  # lower-case; compared against suffix.lower()
GROUPS = ("CONTROL", "ID")  # CONTROL = healthy control, ID = patient
HANDS = ("left", "right")

_UPDRS_RE = re.compile(r"[0-4]")
_MAX_SHOWN = 10  # longest list of line numbers or hashes put in one error message


class HubuFisError(ValueError):
    """The HUBU-FIS inputs break an integrity rule."""


def pseudonym(clip_or_participant_id: object) -> str:
    """Short hash for printed lines. Never print a raw HUBU-FIS ID.

    HUBU-FIS IDs are already dataset pseudonyms, and the ID space is tiny
    (CONTROL1..n, ID1..n), so this 8-character hash can be reversed by trying
    every ID. It follows the repo convention (scripts/count_uci470.py); it is
    not anonymisation.
    """
    return hashlib.sha256(str(clip_or_participant_id).encode()).hexdigest()[:8]


def _shown(items: list[str]) -> str:
    more = f" (+{len(items) - _MAX_SHOWN} more)" if len(items) > _MAX_SHOWN else ""
    return ", ".join(items[:_MAX_SHOWN]) + more


def load_labels(csv_path: Path) -> pd.DataFrame:
    """Read and check the label CSV.

    Returns one row per clip with columns ``ID``, ``UPDRS`` (int), ``participant``,
    ``group``, ``hand`` and ``line`` (1-based line number in the CSV file).
    """
    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if tuple(table.columns) != LABEL_COLUMNS:
        raise HubuFisError(
            f"label CSV columns must be exactly {list(LABEL_COLUMNS)}, found {list(table.columns)}"
        )
    table["line"] = range(2, len(table) + 2)  # line 1 is the header
    if table.empty:
        raise HubuFisError("label CSV has no rows")

    bad_updrs = table.loc[~table["UPDRS"].map(lambda v: bool(_UPDRS_RE.fullmatch(v))), "line"]
    if len(bad_updrs):
        raise HubuFisError(
            f"{len(bad_updrs)} rows have a blank or invalid UPDRS (must be 0-4), "
            f"CSV lines: {_shown([str(n) for n in bad_updrs])}"
        )
    table["UPDRS"] = table["UPDRS"].astype(int)

    keys, bad_ids = [], []
    for clip, line in zip(table["ID"], table["line"], strict=True):
        try:
            keys.append(parse_clip_id(clip))
        except UnparseableClipId:
            bad_ids.append(str(line))
    if bad_ids:
        raise HubuFisError(
            f"{len(bad_ids)} IDs do not match <CONTROL|ID><number>_<DCHA|IZDA> "
            f"(whitespace counts as an error), CSV lines: {_shown(bad_ids)}"
        )

    duplicated = table.loc[table["ID"].duplicated(keep=False), "line"]
    if len(duplicated):
        raise HubuFisError(
            f"{len(duplicated)} rows share an ID with another row, CSV lines: "
            f"{_shown([str(n) for n in duplicated])}"
        )

    table["participant"] = [k.participant for k in keys]
    table["group"] = [k.group for k in keys]
    table["hand"] = [k.hand for k in keys]
    table["number"] = [k.number for k in keys]

    spellings = table.groupby(["group", "number"])["participant"].nunique()
    clashes = spellings[spellings > 1]
    if len(clashes):
        clashing = pd.MultiIndex.from_frame(table[["group", "number"]]).isin(clashes.index)
        names = sorted({pseudonym(p) for p in table.loc[clashing, "participant"]})
        raise HubuFisError(
            f"{len(clashes)} participant numbers are written in more than one way "
            f"(e.g. 1 and 01), hashed participants: {_shown(names)}"
        )
    return table.drop(columns="number")[["ID", "UPDRS", "participant", "group", "hand", "line"]]


def list_videos(video_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    """Map each video stem to its file. Returns (videos, ignored non-video files).

    Raises if two video files share a stem, because then one clip would have
    two candidate files and which one is used would depend on directory order.
    """
    videos: dict[str, Path] = {}
    ignored: list[Path] = []
    clashes: list[str] = []
    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            ignored.append(path)
            continue
        if path.stem in videos:
            clashes.append(pseudonym(path.stem))
        videos[path.stem] = path
    if clashes:
        raise HubuFisError(
            f"{len(clashes)} clips have more than one video file (same name, different "
            f"suffix), hashed clip IDs: {_shown(sorted(clashes))}"
        )
    return videos, ignored


def check_match(labels: pd.DataFrame, videos: dict[str, Path]) -> None:
    """Raise unless every labelled clip has a video and every video has a label."""
    labelled, on_disk = set(labels["ID"]), set(videos)
    no_video = sorted(pseudonym(c) for c in labelled - on_disk)
    no_label = sorted(pseudonym(c) for c in on_disk - labelled)
    if no_video or no_label:
        raise HubuFisError(
            "label CSV and video folder disagree:\n"
            f"  labelled but no video ({len(no_video)}): {_shown(no_video)}\n"
            f"  video but no label    ({len(no_label)}): {_shown(no_label)}\n"
            "  (IDs are hashed; compare with hubu_fis.pseudonym(<clip ID>))"
        )


def participant_table(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per participant, sorted by participant key.

    Columns: ``group``, ``has_left``, ``has_right``, ``updrs_left``,
    ``updrs_right``, ``abs_diff_rl`` and ``updrs_max``. A missing hand gives
    NA (pandas nullable Int64), never 0, and so does its ``abs_diff_rl``.

    This is the table later tasks group cross-validation folds on, so that the
    two hands of one person always land on the same side of a split.
    """
    if labels.duplicated(["participant", "hand"]).any():
        raise HubuFisError("a participant has more than one clip for the same hand")
    if (labels.groupby("participant")["group"].nunique() > 1).any():
        raise HubuFisError("a participant's clips carry different groups")

    wide = labels.pivot(index="participant", columns="hand", values="UPDRS")
    wide = wide.reindex(columns=list(HANDS)).astype("Int64")
    table = pd.DataFrame(
        {
            "group": labels.groupby("participant")["group"].first(),
            "has_left": wide["left"].notna(),
            "has_right": wide["right"].notna(),
            "updrs_left": wide["left"],
            "updrs_right": wide["right"],
            "abs_diff_rl": (wide["right"] - wide["left"]).abs(),
            "updrs_max": wide.max(axis=1, skipna=True).astype("Int64"),
        }
    )
    table.index.name = "participant"
    return table.sort_index()
