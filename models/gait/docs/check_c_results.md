# Check C results: MediaPipe on KOA-PD-NM (GAIT-09)

Run date: 2026-10-08. Model: Pose Landmarker heavy, video mode, full frame (no crop).
All clips are 1920x1080, filmed side-on at about 8 m. Clips are named by role and a short hash
only (no file names, no subject IDs). Two sets of three clips (healthy, mild-PD, severe-PD).
Set B was chosen after set A failed. Both are reported, and neither replaces the other.

## Pass mark

- Plan: person found in >= 90% of all frames, and >= 2 gait cycles in the world ankle trace.
- Confirmed on 2026-10-08: cycles are counted on **clean frames** only, meaning the
  person is found and >= 80% of the 10 leg landmarks are visible, and the clip needs >= 3 s of
  clean frames. Reason: MediaPipe can "find" a person while seeing almost no legs, and the ankle
  traces in those frames produce false steps. The 80% and 3 s values were chosen before these
  results and not tuned on them.
- "In view" is the detection rate counted only from the first to the last found frame. It is a
  diagnostic column, not a pass mark.

## Set A

| role | id | fps | sec | found | first found | in view | clean (s) | cycles L/R (clean) | cycles (no filter) | image L/R | plan's mark | in-view |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| healthy | w-9cb231d9 | 30 | 3.9 | 75% | 0.9 s | 96% | 2.8 | 1 / 2 | 2 | 1 / 2 | FAIL | FAIL |
| mild | w-aacff792 | 50 | 15.5 | 79% | 3.0 s | 97% | 10.9 | 4 / 4 | 7 | 5 / 4 | FAIL | PASS |
| severe | w-61e7768d | 50 | 20.2 | 94% | 1.2 s | 100% | 17.9 | 20 / 16 | 19 | 7 / 11 | PASS | PASS |

## Set B

| role | id | fps | sec | found | first found | in view | clean (s) | cycles L/R (clean) | cycles (no filter) | image L/R | plan's mark | in-view |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| healthy | w-f698b9e3 | 30 | 12.9 | 92% | 1.1 s | 100% | 11.4 | 8 / 11 | 11 | 10 / 9 | PASS | PASS |
| mild | w-32869914 | 30 | 18.5 | 97% | 0.3 s | 99% | 12.3 | 7 / 6 | 6 | 3 / 8 | PASS | PASS |
| severe | w-828c0e49 | 50 | 18.5 | 99% | 0.0 s | 99% | 15.1 | 8 / 9 | 12 | 3 / 6 | PASS | PASS |

Whole-clip result: set A 1/3, set B 3/3. Counted from when the person is found: set A 2/3
(healthy fails on 2.8 s of clean tracking, 0.2 s short of the 3 s rule), set B 3/3.

## What the plots show

- **Why set B scored higher than set A.** The 90% test divides found frames by all frames, so
  walk-in time counts against a clip. Set A: first found at 0.9 s and 3.0 s, healthy only 3.9 s
  long. Set B: first found at 1.1 s, 0.3 s and 0.0 s, clips of 13 to 19 s. Once found, tracking
  was equally good in both sets (in view 96 to 100%).
- **The clean-frame filter removes the junk.** Both mild clips show the same pattern: found, but
  legs visible in only about 20% of frames until about 4.4 s, with ankle heights jumping around.
  The filter excludes that stretch, and the step markers no longer appear in it.
- **Healthy B is the one clearly trustworthy clip:** found from 1.1 s, legs visible nearly
  throughout, regular steps in both traces, 11.4 s clean, 8 to 11 cycles.
- **Mild A is the next best:** 10.9 s clean, 4 cycles per foot at regular intervals of roughly
  1.7 to 2 s, world and image counts agree (4/4 and 5/4). It fails only the 90% detection test,
  because of the late entry.
- **Mild B is low confidence even though it passes.** The clean stretch (about 4.5 to 17.8 s)
  has only a few centimetres of ankle swing, and the image trace is nearly flat. World and image
  counts disagree (7/6 against 3/8). The cycle count rose from 6 to 7 after filtering because
  removing the junk narrowed the trace's range, which lowered the relative threshold for what
  counts as a step. There is also a one-frame spike at about 18 s when the person leaves the
  frame. Treat this clip's cycle count as unverified.
- **Severe clips: tracked, not countable.** Both are found throughout with small, irregular
  ankle swings, and world and image counts still disagree (20/16 against 7/11; 8/9 against 3/6).
  Filtering dropped severe B from 12 to 9 cycles. A severe PASS means a person was tracked, not
  that steps were counted correctly.

## Conclusions

1. **Verdict: conditional pass.** MediaPipe finds and follows the walker at 8 m, and with the
   clean-frame rule the junk stretches no longer inflate the counts. Step counting from ankle
   height alone is only trustworthy on the healthy walker and on mild A. Do not rely on it for
   mild or severe gait features.
2. Step detection for the gait features (GAIT-16 onwards) needs a better method than counting
   ankle-height minima, and must be checked per clip against a plot.
3. 8 m is outside the model card's intended range (people beyond about 4 m are out of scope).
   These clips show that the pipeline runs. They say nothing about gait-feature accuracy.
   The demo video should be recorded at 2 to 4 m, side-on, with the whole walk in frame from the
   first frame and at least 10 s long.
4. The follow-crop option (`--follow`) was worse on all three set A clips (66%, 85%, 72% found)
   and is not recommended.
5. KOA-PD-NM stays demo-only. It supplies no UPDRS labels (severity scale and rater undocumented).

## Limits

Six clips, one walker each. Set B was run after set A failed, so the pass count flatters the
method a little. The thresholds (90%, 80%, 3 s, 2 cycles) are rules of thumb, and set A healthy
fails the clean-time rule by 0.2 s. No claim is made about MediaPipe accuracy across patients.
Cycle counts come from ankle height only and are a rough sanity check, not a gait feature.
Plots and JSON are in `models/gait/artifacts/check_c_setA` and `check_c_setB` (git-ignored).

## Open decisions

- [x] Plan owner confirms the clean-frame rule (80% leg visibility, 3 s) as part of Check C.
  Confirmed by Nethal on 2026-10-08.
