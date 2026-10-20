import type { GaitResult } from "./gaitTypes";

// Three real `gait-result/1` records from the KOA-PD-NM run (python models/gait/src/run_koa_pd_nm.py),
// with the clip name dropped. They stand in for the backend until the screen is connected
// (GAIT-31): a clear score, a mixed score with a stride warning, and a video that was not scored.
// Only the CLEAR one has the one-person box ticked.

export const MOCK_GAIT_SCORED_CLEAR: GaitResult = {
  "modality": "gait",
  "schema_version": "gait-result/1",
  "available": true,
  "video": "w-e40cda55",
  "reasons": [],
  "warnings": [
    "You confirmed one person is walking. The video itself was not checked for other people."
  ],
  "gate": {
    "passed": true,
    "reasons": [],
    "warnings": [
      "You confirmed one person is walking. The video itself was not checked for other people."
    ],
    "checks": {
      "frame_rate": {
        "value": 50.0,
        "limit": 20.0,
        "ok": true
      },
      "leg_visibility": {
        "value": 0.9738067520372526,
        "limit": 0.8,
        "ok": true
      },
      "filled_share": {
        "value": 0.015133876600698487,
        "limit": 0.2,
        "ok": true
      },
      "facing": {
        "value": 1.0,
        "limit": 0.9,
        "ok": true
      },
      "strides": {
        "value": 18.0,
        "limit": 3.0,
        "ok": true
      }
    }
  },
  "walk": {
    "seconds": 17.18,
    "start_s": 2.62,
    "end_s": 19.8,
    "fps": 50.0
  },
  "predicted_class": 2,
  "predicted_label": "class 2-3",
  "class_labels": [
    "class 0",
    "class 1",
    "class 2-3"
  ],
  "probabilities": [
    0.001788115012459457,
    0.07498174905776978,
    0.9232301115989685
  ],
  "probabilities_calibrated": false,
  "expected_severity": 1.9214419722557068,
  "expected_severity_note": "A summary of the three model scores on a 0 to 2 scale; not a UPDRS score.",
  "top_features": [
    {
      "name": "hip_flexion_range_deg",
      "value": 19.81311,
      "shap": 1.5717421770095825,
      "direction": "towards",
      "strength": "strongly",
      "sentence": "The model used hip swing range (20\u00b0), lower than about 9 in 10 of the training walks, which strongly pushed its score towards a moderate-to-severe impairment pattern."
    },
    {
      "name": "knee_flexion_range_deg",
      "value": 24.009143,
      "shap": 1.2331421375274658,
      "direction": "towards",
      "strength": "strongly",
      "sentence": "The model used knee bend range (24\u00b0), lower than about 9 in 10 of the training walks, which strongly pushed its score towards a moderate-to-severe impairment pattern (its effect varies with the other inputs)."
    },
    {
      "name": "step_length_norm",
      "value": 0.125861,
      "shap": -0.3372251093387604,
      "direction": "away",
      "strength": "moderately",
      "sentence": "The model used step length (0.13 leg lengths), lower than about 9 in 10 of the training walks, which moderately pushed its score away from a moderate-to-severe impairment pattern (its effect varies with the other inputs)."
    }
  ],
  "feature_vector": {
    "stride_time_mean": 1.728889,
    "cadence": 69.40874,
    "stride_time_cv": 39.764547,
    "step_length_norm": 0.125861,
    "step_length_asymmetry": 0.696793,
    "knee_flexion_range_deg": 24.009143,
    "hip_flexion_range_deg": 19.81311,
    "arm_swing_range_norm": 0.0975
  },
  "model_version": "gait-xgb-ed1c5ad4",
  "one_person_confirmed": true
};

export const MOCK_GAIT_SCORED_MIXED: GaitResult = {
  "modality": "gait",
  "schema_version": "gait-result/1",
  "available": true,
  "video": "w-9ccd2ff3",
  "reasons": [],
  "warnings": [
    "One person in view is assumed: the video was not checked for other people.",
    "left and right stride times differ by 49% (over 25%): one foot may have extra or missed heel strikes"
  ],
  "gate": {
    "passed": true,
    "reasons": [],
    "warnings": [
      "One person in view is assumed: the video was not checked for other people."
    ],
    "checks": {
      "frame_rate": {
        "value": 50.0,
        "limit": 20.0,
        "ok": true
      },
      "leg_visibility": {
        "value": 0.9866666666666667,
        "limit": 0.8,
        "ok": true
      },
      "filled_share": {
        "value": 0.0,
        "limit": 0.2,
        "ok": true
      },
      "facing": {
        "value": 0.9866666666666667,
        "limit": 0.9,
        "ok": true
      },
      "strides": {
        "value": 7.0,
        "limit": 3.0,
        "ok": true
      }
    }
  },
  "walk": {
    "seconds": 4.5,
    "start_s": 1.26,
    "end_s": 5.76,
    "fps": 50.0
  },
  "predicted_class": 2,
  "predicted_label": "class 2-3",
  "class_labels": [
    "class 0",
    "class 1",
    "class 2-3"
  ],
  "probabilities": [
    0.4017150402069092,
    0.11836464703083038,
    0.47992032766342163
  ],
  "probabilities_calibrated": false,
  "expected_severity": 1.0782053023576736,
  "expected_severity_note": "A summary of the three model scores on a 0 to 2 scale; not a UPDRS score.",
  "top_features": [
    {
      "name": "cadence",
      "value": 134.185304,
      "shap": 0.6734698414802551,
      "direction": "towards",
      "strength": "strongly",
      "sentence": "The model used walking pace (134 steps/min), higher than about 9 in 10 of the training walks, which strongly pushed its score towards a moderate-to-severe impairment pattern."
    },
    {
      "name": "stride_time_cv",
      "value": 26.094773,
      "shap": 0.38376304507255554,
      "direction": "towards",
      "strength": "moderately",
      "sentence": "The model used stride-time variability (26.1%), higher than about 9 in 10 of the training walks, which moderately pushed its score towards a moderate-to-severe impairment pattern."
    },
    {
      "name": "stride_time_mean",
      "value": 0.894286,
      "shap": 0.3279384970664978,
      "direction": "towards",
      "strength": "moderately",
      "sentence": "The model used stride time (0.89 s), lower than about 9 in 10 of the training walks, which moderately pushed its score towards a moderate-to-severe impairment pattern (its effect varies with the other inputs)."
    }
  ],
  "feature_vector": {
    "stride_time_mean": 0.894286,
    "cadence": 134.185304,
    "stride_time_cv": 26.094773,
    "step_length_norm": 0.241669,
    "step_length_asymmetry": 2.0,
    "knee_flexion_range_deg": 48.565986,
    "hip_flexion_range_deg": 31.134912,
    "arm_swing_range_norm": 0.193305
  },
  "model_version": "gait-xgb-ed1c5ad4",
  "one_person_confirmed": false
};

export const MOCK_GAIT_NOT_SCORED: GaitResult = {
  "modality": "gait",
  "schema_version": "gait-result/1",
  "available": false,
  "video": "w-b44bd999",
  "reasons": [
    "The walk has 2 real strides on both feet; at least 3 are needed."
  ],
  "warnings": [
    "One person in view is assumed: the video was not checked for other people."
  ],
  "gate": {
    "passed": false,
    "reasons": [
      "The walk has 2 real strides on both feet; at least 3 are needed."
    ],
    "warnings": [
      "One person in view is assumed: the video was not checked for other people."
    ],
    "checks": {
      "frame_rate": {
        "value": 50.0,
        "limit": 20.0,
        "ok": true
      },
      "leg_visibility": {
        "value": 0.9654205607476636,
        "limit": 0.8,
        "ok": true
      },
      "filled_share": {
        "value": 0.009345794392523364,
        "limit": 0.2,
        "ok": true
      },
      "facing": {
        "value": 1.0,
        "limit": 0.9,
        "ok": true
      },
      "strides": {
        "value": 2.0,
        "limit": 3.0,
        "ok": false
      }
    }
  },
  "walk": {
    "seconds": 2.14,
    "start_s": 8.52,
    "end_s": 10.66,
    "fps": 50.0
  },
  "predicted_class": null,
  "predicted_label": null,
  "class_labels": [
    "class 0",
    "class 1",
    "class 2-3"
  ],
  "probabilities": null,
  "probabilities_calibrated": false,
  "expected_severity": null,
  "expected_severity_note": "A summary of the three model scores on a 0 to 2 scale; not a UPDRS score.",
  "top_features": [],
  "feature_vector": null,
  "model_version": "gait-xgb-ed1c5ad4",
  "one_person_confirmed": false
};

export const MOCK_GAIT_RESULTS: readonly { key: string; label: string; result: GaitResult }[] = [
  { key: "clear", label: "Clear score", result: MOCK_GAIT_SCORED_CLEAR },
  { key: "mixed", label: "Mixed score", result: MOCK_GAIT_SCORED_MIXED },
  { key: "none", label: "Not scored", result: MOCK_GAIT_NOT_SCORED },
];
