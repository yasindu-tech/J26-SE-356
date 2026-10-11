# models/tapping — finger tapping

**Owner:** Weerasinghe P.N.M. (IT23247918)

**Question.** Can smartphone-video finger tapping recover the kinematics
MDS-UPDRS Part III item 3.4 rates, well enough to contribute to screening?

## Data — HUBU-FIS (Zenodo, DOI 10.5281/zenodo.17738775)
Raw files live in `data/raw/HUBU-FIS/` (`fis_diagnostic.csv` and `videos/`) and
are never committed. Every number below is printed by
`python scripts/count_hubu_fis.py --probe-video`; the full tables are in
[docs/hubu_fis_counts.md](docs/hubu_fis_counts.md) (machine-readable:
`docs/hubu_fis_counts.json`).

| Metric | Value |
|---|---|
| Clips | 234 (226 .mp4, 8 .MOV), labels and videos match 1:1 |
| Participants | 118 (43 CONTROL, 75 ID = patient); 116 with both hands |
| UPDRS 3.4 rating per clip, 0/1/2/3 | 71 / 99 / 38 / 26 (4 never used) |
| CONTROL vs ID | 85 / 149 clips (1.75:1); 43 / 75 participants (1.74:1) |
| UPDRS ≥ 1 vs 0 | 163 / 71 clips (2.30:1) |

- **UPDRS here is a clinician's rating of one hand's tapping, not a confirmation
  of disease**: 27 control clips are rated 1 and 13 patient clips are rated 0. It
  may be a label or a stratification variable, never a model input.
- **Both hands of a person share one participant key** (`src/clip_id.py`); folds
  must be grouped on it (`hubu_fis.participant_table`).
- **Device confound.** Some recording settings occur in only one group: all 8
  .MOV clips and the 352x640 / 640x352 clips are CONTROL; the 1080x2400,
  1280x720 and 60 fps clips are patients. Plan: per-clip QC logging, a
  sensitivity analysis without these clips, and resolution-independent features.
- **Orientation.** 193 clips are stored rotated with 90° metadata. Sizes are
  reported as displayed (after rotation), which is what the landmark model sees.
- **Keep the probe.** A run without `--probe-video` refuses to overwrite counts
  that contain the confound tables (exit 1, nothing written); pass
  `--allow-drop-probe` only if dropping them is intended.

## Pipeline
Video → MediaPipe Hands → landmark tracking → tap detection → kinematics
(rate, amplitude, decrement/fatigue, hesitations, arrhythmicity) → model → SHAP.

## Hand landmarks (FT-11)
`src/hand_landmarks.py` runs the MediaPipe HandLandmarker in VIDEO mode (`num_hands=1`) over
one clip and writes one row per frame to `data/interim/landmarks/` (gitignored): 21 landmarks
as normalised, pixel (displayed frame) and world coordinates. Download the model first with
`models/tapping/fetch_model.sh`.

```bash
python models/tapping/src/hand_landmarks.py --video path/to/clip.mp4
```

- A frame with no hand has `detected = False` and NaN values, never zero.
- `timestamp_ms` is the integer clock given to MediaPipe. **Tap timing must use `pts_ms`**,
  the container's own time: phone clips are not constant frame rate.
- fps, width, height, library versions, the model hash and the detector settings are stored
  once in the parquet metadata (`read_clip_geometry`, `read_metadata`).
- The handedness label is informational only: take the hand from the clip name (DCHA / IZDA).
  On 2 of the 4 clips checked in FT-11 the label flickered between Left and Right while the
  same hand was tracked correctly.
- Exit codes: 0 ok, 1 unreadable clip, 2 missing video or model.

Known limitations (from looking at ten full-resolution crops of two clips; eyeballed, not
measured against ground truth):

- **Absolute pixel positions are approximate. Normalise distances by hand size** (for example
  wrist 0 to knuckle 9), and do not read single-frame positions too literally.
- Landmarks 4 and 8 sit on the fingertip but are usually a little short of the visible nail
  end, always inboard along the finger. The miss varies from frame to frame (about a tenth of
  a hand size is typical; it was larger, about a third, at frame 0 of one clip).
- In the same clips, the VIDEO-mode and the single-image result for one frame differed by up
  to about a fifth of a hand size per landmark (frame 0 is identical in both modes).

## Raw distance plot (FT-12)
`src/plot_distance.py` plots the raw distance between thumb tip (landmark 4) and index tip
(landmark 8), in pixels of the displayed frame, for one clip. It needs the landmarks file
from FT-11 first. The plot goes to `data/interim/plots/` (gitignored) and is named by the
clip's 8-character hash only; titles and axis text never contain the raw clip ID.

```bash
python models/tapping/src/plot_distance.py --clip <8-character hash>
python models/tapping/src/plot_distance.py --parquet path/to/landmarks.parquet --compare-clocks
```

- **What it shows:** the raw curve against `pts_ms` (the container's time). A frame with no
  hand breaks the line and gets a red tick along the bottom edge; nothing is drawn at zero.
  The median hand size (wrist 0 to knuckle 9) is a dashed reference line and a labelled
  right-hand axis; it never replaces the pixel signal. `--compare-clocks` also plots the
  signal against the index/fps clock, with a zoom where the two clocks differ most.
- **Printed numbers:** duration, min / median / max distance in pixels and in hand sizes
  (reference only), frames without a distance, and a rough open/close cycle count.
- **Limits:** the signal is raw, with no smoothing, no gap filling and no peak detection
  (FT-13, FT-19, FT-20). The cycle count is a simple crossing count (below 25%, then above
  50% of the clip's own 5th to 95th percentile range), **not the tap detector**, and it
  depends on those thresholds. Pixel positions are approximate (see the FT-11 limitations).
  At 30 fps a fast open or close edge gets only 1 to 3 samples, so edge timing and speed are
  coarse on fast clips.
- **Checked on four clips** (FT-12): all four show a repeating open/close oscillation of
  about 0.5 to 1.4 cycles per second, with no missing frames. Reproduce with the command above.

## Hand-size normalisation (FT-13)
`src/hand_scale.py` divides the thumb-index distance by a length of the hand itself, so the
signal no longer grows when the hand moves toward the camera.

```python
result = normalised_distance(df, scale="w9", mode="rolling_median", window_s=3.0)
result.values  # one value per frame, in hand sizes; NaN where there is no distance or scale
result.metadata()  # scale, mode, window, rejected frames: store this next to the signal
```

```bash
python models/tapping/src/hand_scale.py --clip <8-character hash>   # plot in data/interim/plots
```

- **Decision: scale `w9` (wrist 0 to knuckle 9), mode `rolling_median` over a centred 3 s
  window of `pts_ms`.** Runner-up: `median4` (median of the four wrist-to-knuckle lengths 5, 9,
  13, 17), which was indistinguishable on every criterion and is the fallback if single knuckle
  errors turn out to matter. The decision used physical criteria only, never UPDRS or
  CONTROL/ID.
- **Why not per frame:** the per-frame scale moves with the tap itself (after removing slow
  drift, median |r| with the raw distance 0.41 for `w9`, over 35% of 42 clips above 0.5), so
  dividing by it would shrink the tap amplitude. A rolling median averages that out (median
  |r| 0.07, 5% of clips above 0.5). **Why not one number per clip:** it cannot follow an
  approach; on a synthetic zoom from 0.85x to 1.1x it leaves a 6% median error, the rolling
  median 0.3%.
- **Why not palm width (MCP5 to MCP17):** it is the noisiest candidate (frame-to-frame jitter 17%,
  against 2.5% for `w9`) and co-varies most with the tap.
- **Guard:** a frame scale that is not finite, not positive, or below 25% of the clip's own
  median scale is rejected (NaN) and counted. NaN stays NaN; nothing is zero-filled.
  The scale uses only landmarks 0, 5, 9, 13 and 17, never the thumb or index tip.
- **Evidence** (42 clips tracked with `hand_landmarks.py`, chosen by resolution and
  orientation and not by label): the strongest real approach or recession is about 25%
  hand-size change over a clip. The script column `camera_trend_removed` is (raw open-level trend minus normalised
  open-level trend) divided by the hand-size change, where a trend is the change over the
  clip of the 90th percentile in 4 s blocks, as a fraction of its mean. It measures how much
  the normalisation shifted the trend, not how clean the remainder is. On the four clearest
  approach/recede clips it shifted the open-level trend by roughly the size of the hand-size
  change (80% to 128%), which confirms the correction has the expected size but does not
  show that the remaining trend is free of camera effect; the main evidence is the synthetic
  zoom test and the plots. On two of these clips the normalised trend is larger than the raw
  one because a real decrement is revealed: dcd528b5 (raw -0.095, normalised -0.293) and
  11054d82 (raw -0.031, normalised -0.188). The
  world landmarks put the open aperture at 6.7 to 16.6 cm across the 42 clips, so the
  differences between clips are not only camera distance. Reproduce every number in this
  section with `python scripts/evaluate_hand_scale.py` (reads the existing parquet files,
  writes to `data/interim/hand_scale_eval/`, shows clips by hash only).
- **Residual coupling:** with the 3 s window, 2 of 42 clips still have |r| > 0.5 between the
  scale and the tap (tap rates 1.2 and 1.5 per second, so it is not a slow-tapper effect, and
  longer windows do not reliably clear it: one clip drops under 0.2 at 5 s and is back at 0.6 at 8 s). FT-15 should flag such clips (|r| > 0.5) rather than drop
  them.
- **Limits:** normalisation cannot fix rotation of the hand out of the image plane
  (foreshortening shrinks `w9` and inflates the normalised aperture; the 2D open aperture was 1.07 to
  1.91 times (median 1.33) the world-landmark value across the 42 clips); the closed floor is never zero
  (about 0.1 hand sizes); the initial open plateau and landmark noise stay; and the window is
  centred, so it uses frames after the current one. That is fine for recorded clips; live
  capture would need a causal window. A genuine
  amplitude decrement is not removed: on one approach clip the raw peaks looked flat while the
  hand grew, and the normalised peaks showed the decrement.
- **World landmarks** are a model prediction with a hand-shape prior (`w9` is 9 to 10 cm in
  almost every clip), so they are a reference for pose, not a measurement of hand size.

## Overlay GIF (FT-14)
`src/overlay_gif.py` makes a short GIF of the tracked hand skeleton on one clip, cropped to the
hand, with the normalised thumb-index distance (FT-13) as a live trace underneath. A poster
still (PNG) is saved from the same window. It is a **proof-of-concept visual, not a result**.

```bash
python models/tapping/src/overlay_gif.py --clip <8-character hash> --start-s 3 --duration-s 4
# options: --width 480 --fps 15 --blur-outside --out-dir data/interim/gifs
```

- **Needs** the landmarks parquet (FT-11) and the video in `data/raw/HUBU-FIS/videos/`.
  Output: `data/interim/gifs/<hash>_overlay.gif` and `<hash>_overlay_poster.png` (gitignored).
- **Shows:** bones and joints of the 21 landmarks, thumb tip (4, orange) and index tip (8,
  magenta) highlighted and joined by a thin line, and the trace with a cursor that moves with
  the video. A frame without a hand shows the picture and no skeleton.
- **Timing follows `pts_ms`.** The window is chosen by `pts_ms`; with `--fps` below the
  clip's rate every n-th frame is kept (15 fps from a 30 fps clip). GIF delays come in 10 ms
  units, so each delay is the gap to the next kept frame rounded on a running total: at 15 fps
  from a 30 fps clip they alternate 60 and 70 ms, the total stays within a few ms of the `pts_ms` span, and
  irregular gaps are kept. Rates with exact delays are 50, 25, 20 and 10 fps.
- **Limits, enforced:** width 160 to 640 px (default 480), at most 6 s, 150 frames and 5 MB;
  one shared 72-colour palette (the overlay colours are reserved so the tips stay visible),
  no dithering, looping forever. The output folder must be inside `data/interim`.
- **Every decoded frame's container time is checked against the parquet's `pts_ms`**, and
  auto-rotation is switched on and read back (as in FT-11), so a shifted frame or a rotated
  picture is an error, not a silent mistake.
- **Privacy:** nothing is drawn but the skeleton and the trace: no rating, no group, no clip
  ID, no wording about disease. Files are named by the clip's hash only. The picture is
  cropped to the hand; `--blur-outside` also blurs everything but the hand. **Look at the
  result before showing it in public**, and do not copy it out of `data/interim` for anything
  but the slide.
- **Limits:** landmarks 4 and 8 sit a little short of the visible nail (FT-11), so the markers
  touch before the nails do; the colour palette flattens skin tones; one clip, one window,
  no validation.

## Known problems
- Tapping features are **derived from the same behaviour** the MDS-UPDRS item
  rates. This is acceptable — raw kinematics, not a clinician's score — but it
  must be stated explicitly rather than discovered by a reviewer.
- Decrement and hesitation are the clinically distinctive signals; raw tap rate
  alone is weakly specific.
- Hand visibility and framing are the dominant capture-quality failure modes —
  gate on device.
