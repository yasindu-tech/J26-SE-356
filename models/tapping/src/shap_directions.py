"""Per-feature SHAP direction table (task FT-46, code part). Direction describes how the model's
score moves with a feature value, not cause and not a clinical finding. Reads only
data/processed/xai/<target>/<set>/ (no refit, no labels): Spearman of value against mean SHAP per
clip, tertile means and missing-value means per feature. Exit codes: 0 ok, 1 guard failure,
2 missing input or a path outside the xai folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv_splits as cvs
import numpy as np
import pandas as pd
import scipy
import shap_figures as sf
import treeshap_heldout as tsh
from cv_splits import GuardFailure, InputProblem
from scipy.stats import spearmanr

TOOL = "shap-directions-1"
CONFIGS = (("ge2", "A"), ("ge1", "A"), ("ge2", "B"))  # pre-registered; model is lgbm only
RHO_CUT = 0.30
MIN_TERTILE_CLIPS = 9
OUT_FILE, META_FILE = "shap_directions.csv", "shap_directions_meta.json"
COLUMNS = [
    "feature", "global_rank", "mean_abs_shap", "n_clips_value", "n_clips_nan",
    "rho_value_vs_shap", "direction", "mean_shap_value_low_tertile",
    "mean_shap_value_high_tertile", "mean_shap_nan", "mean_shap_non_nan", "share_positive_shap",
]  # fmt: skip


def spearman(value: np.ndarray, shap: np.ndarray) -> float:
    """Spearman rho (average ranks); NaN, never 0, when either side is constant or too short."""
    if len(value) < 2 or np.ptp(value) == 0 or np.ptp(shap) == 0:
        return float("nan")
    return float(spearmanr(value, shap).statistic)


def direction_of(rho: float) -> str:
    if np.isnan(rho):
        return ""
    if rho >= RHO_CUT:
        return "up"
    return "down" if rho <= -RHO_CUT else "unclear"


def tertile_means(value: np.ndarray, shap: np.ndarray) -> tuple[float, float]:
    """Mean SHAP of the low and high tertile of the values; ties at a cut go to the lower tertile."""
    if len(value) < MIN_TERTILE_CLIPS:
        return float("nan"), float("nan")
    cut_low, cut_high = np.quantile(value, [1 / 3, 2 / 3])
    low, high = shap[value <= cut_low], shap[value > cut_high]
    return (float(low.mean()) if len(low) else float("nan"),
            float(high.mean()) if len(high) else float("nan"))  # fmt: skip


def direction_row(
    feature: str, rank: int, mean_abs: float, value: np.ndarray, shap: np.ndarray
) -> dict[str, Any]:
    value, shap = np.asarray(value, dtype=float), np.asarray(shap, dtype=float)
    present = ~np.isnan(value)
    rho = spearman(value[present], shap[present])
    low, high = tertile_means(value[present], shap[present])
    any_nan = bool((~present).any())
    return {
        "feature": feature, "global_rank": rank, "mean_abs_shap": mean_abs,
        "n_clips_value": int(present.sum()), "n_clips_nan": int((~present).sum()),
        "rho_value_vs_shap": rho, "direction": direction_of(rho),
        "mean_shap_value_low_tertile": low, "mean_shap_value_high_tertile": high,
        "mean_shap_nan": float(shap[~present].mean()) if any_nan else float("nan"),
        "mean_shap_non_nan": float(shap[present].mean()) if any_nan else float("nan"),
        "share_positive_shap": float((shap > 0).mean()),
    }  # fmt: skip


def direction_table(glob: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in glob.sort_values(["rank", "feature"], kind="stable").itertuples(index=False):
        rows.append(direction_row(r.feature, int(r.rank), float(r.mean_abs_shap_mean_over_repeats),
                                  clips[f"value_{r.feature}"].to_numpy(), clips[f"shap_{r.feature}"].to_numpy()))  # fmt: skip
    return pd.DataFrame(rows, columns=COLUMNS)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g", na_rep="").encode()


def run_config(cv_root: Path, xai: Path, target: str, fs: str) -> dict[str, Any]:
    cols = list(tsh.ch.FEATURE_SETS[fs])
    folder = xai / target / fs
    inputs = sf.read_inputs(folder, cols)
    official_path = cv_root / target / f"meta_{tsh.MODEL}_{fs}.json"
    if not official_path.is_file():
        raise InputProblem(f"missing input: official meta for {target} {fs}")
    official = json.loads(official_path.read_text())
    sf.check_meta(inputs["meta"], official, target, fs)
    sf.check_tables(inputs, cols, f"{target} {fs}")
    clips = sf.per_clip_means(inputs["values"], cols)
    table = direction_table(inputs["glob"], clips)
    data = csv_bytes(table)
    out_dir = xai / "directions" / target / fs
    cvs._atomic(out_dir / OUT_FILE, data)
    meta = {
        "tool": TOOL, "target": target, "feature_set": fs, "model": tsh.MODEL, "columns": cols,
        "constants": {"rho_cut": RHO_CUT, "min_tertile_clips": MIN_TERTILE_CLIPS,
                      "tertiles": "np.quantile at 1/3 and 2/3 of the non-NaN values; ties at a cut go to the lower tertile",
                      "value_per_clip": "mean over the 5 repeats of SHAP; feature value as recorded"},  # fmt: skip
        "inputs_sha8": {n: cvs.sha8_of((folder / n).read_bytes()) for n in sf.SHAP_INPUTS},
        "features_sha8": official["features_sha8"], "grid_sha8": official["grid_sha8"],
        "splits_sha8": official["splits_sha8"], "files_sha8": {OUT_FILE: cvs.sha8_of(data)},
        "n_clips": len(clips), "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }  # fmt: skip
    cvs._atomic(out_dir / META_FILE, (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())
    return {"target": target, "fs": fs, "table": table, "sha8": {OUT_FILE: meta["files_sha8"][OUT_FILE],
            META_FILE: cvs.sha8_of((out_dir / META_FILE).read_bytes())}, "n_clips": len(clips)}  # fmt: skip


def num(x: float) -> str:
    return "-" if x != x else f"{x:.4f}"


def render(done: Sequence[dict[str, Any]]) -> str:
    lines = []
    for d in done:
        lines.append(f"== {d['target']} Set {d['fs']}: {len(d['table'])} features, {d['n_clips']} clips "
                     f"(direction = how the score moves with the value, not cause)")  # fmt: skip
        lines.append(f"  {'feature':<22}{'rank':>4}{'mean|SHAP|':>11}{'n':>5}{'nan':>5}{'rho':>9}  {'dir':<8}"
                     f"{'low3':>9}{'high3':>9}{'shap_nan':>10}{'shap_val':>10}{'share+':>8}")  # fmt: skip
        for r in d["table"].itertuples(index=False):
            lines.append(f"  {r.feature:<22}{r.global_rank:>4}{num(r.mean_abs_shap):>11}{r.n_clips_value:>5}"
                         f"{r.n_clips_nan:>5}{num(r.rho_value_vs_shap):>9}  {r.direction or '-':<8}"
                         f"{num(r.mean_shap_value_low_tertile):>9}{num(r.mean_shap_value_high_tertile):>9}"
                         f"{num(r.mean_shap_nan):>10}{num(r.mean_shap_non_nan):>10}{num(r.share_positive_shap):>8}")  # fmt: skip
        lines.append("  sha8 " + "  ".join(f"{k.split('.')[0]} {v}" for k, v in d["sha8"].items()))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("cv-root", "xai-root"):
        ap.add_argument(f"--{name}", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    xai = Path(args.xai_root or tsh.xai_root())
    if not cvs.is_inside(xai, tsh.xai_root()):
        print("error: --xai-root must be inside data/processed/xai", file=sys.stderr)
        return 2
    try:
        done = [run_config(cv_root, xai, t, fs) for t, fs in CONFIGS]
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except GuardFailure as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(render(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
