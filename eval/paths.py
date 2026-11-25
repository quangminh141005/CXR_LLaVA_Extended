"""Output folders shared by the eval scripts (one place to change the layout).

results/
  inference/   raw model outputs: <ds>.jsonl, <ds>_labels_<prompt>.jsonl, run_info.json
  labels/      <ds>_pred_labels<tag>.csv, labeler/ (CheXpert-Labeler I/O), mimic_gt_relabel/
  scores/      scores_<ds><tag>.csv
checks/        one-off sanity experiments: preprocess_check/, quant_check/, labeler_validation/
log/           stdout of runs (tee): setup/, inference/, labeling/, scoring/
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = os.path.join(REPO, "results")
RESULTS_INF = os.path.join(RESULTS, "inference")
RESULTS_LABELS = os.path.join(RESULTS, "labels")
RESULTS_SCORES = os.path.join(RESULTS, "scores")

CHECKS = os.path.join(REPO, "checks")
LOG = os.path.join(REPO, "log")
