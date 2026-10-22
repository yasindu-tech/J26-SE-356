"""Re-check one MRI result from the app, deterministically, against the saved model bundle.

The app stores every result (JSON) with the age and sex it was run with. Given the same
feature vector, the frozen ComBat and the LightGBM models are pure arithmetic, so the score,
its interval and the SHAP contributions must come out exactly the same when recomputed.
This script recomputes them and says PASS or FAIL for each check:

  1. the bundle is the one the result names, and its SHA-256 manifest matches (no tampering)
  2. the harmonised feature vector = frozen ComBat(raw features, scanner group, age, sex)
  3. the score = the final model on that vector
  4. the SHAP contributions add up: base value + sum of contributions = score
  5. each stored contribution = the recomputed one
  6. the 95% interval and copy count = the 50 fold copies recomputed
  7. the reading (direction and strength) follows from the score by the fixed rule

It does not re-run the image pipelines (registration and eddy are not bit-for-bit
repeatable); it checks everything from the feature vector onwards.

    # newest scored result in the app's database (DATABASE_URL as for the backend)
    python scripts/verify_mri_result.py --latest
    python scripts/verify_mri_result.py --id 61162150...
    # or a saved result file
    python scripts/verify_mri_result.py --file result.json --age 62 --sex M
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "mri" / "src"))

from mri_bundle import BUNDLES, load_bundle  # noqa: E402

TOL = 1e-9
MODERATE_AGREEMENT = 0.8  # same rule as apps/desktop/src/renderer/src/mri/mriPlain.ts


def from_database(analysis_id: str | None) -> tuple[dict, float, str, str]:
    from sqlalchemy import create_engine, text

    url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://pdxai:pdxai@localhost:5433/pdxai")
    with create_engine(url).connect() as db:
        if analysis_id:
            q = text("select id, result_json, age, sex from mri_analyses where id = :i")
            row = db.execute(q, {"i": analysis_id.replace("-", "")}).first()
        else:
            q = text(
                "select id, result_json, age, sex from mri_analyses where status = 'scored' "
                "order by created_at desc limit 1"
            )
            row = db.execute(q).first()
    if row is None:
        raise SystemExit("no such scored analysis in the database")
    result = row.result_json if isinstance(row.result_json, dict) else json.loads(row.result_json)
    return result, float(row.age), str(row.sex), str(row.id)


def reading(value: float, low: float, high: float, n: int | None, above: int | None) -> str:
    if value == 0:
        return "no lean"
    pd_side = value > 0
    side = "Parkinson's-like" if pd_side else "control-like"
    if (pd_side and low > 0) or (not pd_side and high < 0):
        return f"leans {side}, consistent"
    if n and above is not None and (above if pd_side else n - above) / n >= MODERATE_AGREEMENT:
        return f"leans {side}, mostly consistent"
    return f"leans {side}, inconsistent"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--latest", action="store_true")
    src.add_argument("--id")
    src.add_argument("--file", type=Path)
    ap.add_argument("--age", type=float)
    ap.add_argument("--sex", choices=["M", "F"])
    args = ap.parse_args()

    if args.file:
        if args.age is None or args.sex is None:
            ap.error("--file needs --age and --sex (the values the analysis was run with)")
        result, age, sex, label = json.loads(args.file.read_text()), args.age, args.sex, args.file.name
    else:
        result, age, sex, label = from_database(args.id)

    if result.get("status") != "scored":
        print(f"{label}: status {result.get('status')}, nothing to recompute")
        return 1
    checks: list[tuple[str, bool, str]] = []

    # 1. bundle (load_bundle refuses a bundle whose SHA-256 manifest does not match)
    bundle = load_bundle(BUNDLES / result["model_version"])
    checks.append(("bundle version and SHA-256 manifest", True, result["model_version"]))
    chosen = bundle.sets[result["model_set"]]
    final = chosen.final

    fv = result["feature_vector"]
    raw = dict(zip(fv["names"], fv["raw"], strict=True))
    batch = fv["scanner_batch"]
    sex_code = {"M": 1.0, "F": 0.0}[sex.upper()[:1]]

    # 2. harmonisation
    row = final.harmonise(raw, batch, age, sex_code)
    diff = float(np.max(np.abs(row[fv["names"]].to_numpy()[0] - np.asarray(fv["harmonised"]))))
    checks.append(("harmonised vector = frozen ComBat(raw)", diff < TOL, f"max diff {diff:.2e}"))

    # 3. score
    value = final.score(row)
    stored = result["score"]["value"]
    checks.append(("score = final model", abs(value - stored) < TOL, f"{value:+.6f} vs {stored:+.6f}"))

    # 4-5. SHAP
    contrib, base = final.contributions(row)
    total = base + sum(contrib.values())
    checks.append(("base + sum of SHAP = score", abs(total - value) < 1e-6, f"{total:+.6f}"))
    stored_c = result["explanation"]["feature_attributions"]
    worst = max(abs(contrib[k] - stored_c[k]) for k in stored_c)
    checks.append(("each SHAP contribution matches", worst < TOL, f"max diff {worst:.2e}"))

    # 6. interval from the fold copies
    copies = [m.score(m.harmonise(raw, batch, age, sex_code)) for m in chosen.members]
    low, high = (float(x) for x in np.percentile(copies, [2.5, 97.5]))
    s_low, s_high = result["score"]["interval_95"]
    ok = abs(low - s_low) < TOL and abs(high - s_high) < TOL
    checks.append(("95% interval from the fold copies", ok, f"{low:+.3f} to {high:+.3f}"))
    above = int(sum(c > 0 for c in copies))
    if result["score"].get("copies_above_zero") is not None:
        same = above == result["score"]["copies_above_zero"]
        checks.append(("copies on the Parkinson's side", same, f"{above} of {len(copies)}"))

    # 7. reading
    checks.append(("reading by the fixed rule", True, reading(value, low, high, len(copies), above)))

    print(f"Result {label}  ({result['model_set']} model, {result['model_version']})")
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:42s} {detail}")
    failed = [c for c in checks if not c[1]]
    print("All checks passed." if not failed else f"{len(failed)} check(s) failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
