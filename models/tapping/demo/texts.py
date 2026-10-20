"""Fixed texts of the demo (tasks FT-47..51), no streamlit import: the disclaimers and the
sentence templates shared by the app and the one-page PDF. Words such as diagnosis or detect
appear only in the fixed disclaimer sentences (DISCLAIMER, SCORE_CAPTION, SHAP_CAPTION).
"""

from __future__ import annotations

TITLE = "Finger-tapping explainer (research demo)"
DISCLAIMER = (
    "Research demo. The score is a model output on a research dataset; it is not a diagnosis "
    "and the model does not detect Parkinson's."
)
MODES = ("Example clip (held-out)", "Upload video")
TARGET_LABELS = {"ge2 (primary)": "ge2", "ge1": "ge1"}
NOT_STORED = "Processed on this computer; the video is not stored."
SCORE_CAPTION = (
    "Model score from 0 to 1. Not a calibrated probability and not a diagnosis. "
    "In cross-validation this model reached AUROC {est:.2f} [{lo:.2f}, {hi:.2f}] for {target}."
)
HELD_OUT = "Held-out: this clip was not used to train the models that scored it."
TRAINED_ALL = "Demo model trained on all research clips."
BAND_EXAMPLE = "held-out scores over 5 repeats"
BAND_UPLOAD = "5-95% of 25 cross-validation models"
SHAP_CAPTION = "Contributions explain the model's score, not the cause of any condition."
FPS_NOTE = (
    "This video is about {fps:.0f} fps. Speed features are measured per second, so a high "
    "frame rate can shift them and the score may be slightly off."
)
RANGE_NOTE = "{names} outside the range of the research clips; treat the score with care."
SECONDARY_NOTE = "ge1 is the secondary study target; ge2 is the primary target."
