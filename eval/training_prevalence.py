"""Extension Exp 1: training-prevalence proxy per finding (x-axis of "sensitivity vs prevalence").

The paper does not publish per-finding prevalence of its training set (~590k images, 7 sources).
Two proxies, each written for every Exp 1 row and every VinDr label:

  VinDr train split   image_labels_train.csv, 15,000 images, 3 radiologists each (labels given
                      independently). Per image: positive by majority (more than half of its
                      radiologists, = 2 of 3) [main], or by any radiologist [variant].
                      A row with several VinDr labels (Rib/Clavicle fracture) = any of them.
  MIMIC-CXR train     official mimic-cxr-2.0.0-chexpert.csv(.gz), studies in the official train split
                      (mimic-cxr-2.0.0-split.csv.gz if found, else all studies minus our test studies).
                      Positive = 1 [main]; 1 or -1 [variant]. Only for rows with a CheXpert class
                      (MIMIC is the only free-text training source -> prior of the report generator).

Output: results/scores/vindr_training_prevalence.csv
Usage (on the server, repo root, env cxreval):
    python -u eval/training_prevalence.py 2>&1 | tee log/scoring/training_prevalence.log
"""
import argparse
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from paths import RESULTS_SCORES  # noqa: E402
from validate_labeler import MIMIC_LABELS  # noqa: E402
import vindr_common as vc  # noqa: E402


def vindr_train(path):
    lab = pd.read_csv(path)
    vlabels = [c for c in lab.columns if c not in vc.NON_LABEL_COLS]
    n_rads = lab.groupby("image_id").size()
    print(f"VinDr train: {len(lab)} rows, {len(n_rads)} images, radiologists per image "
          f"{n_rads.value_counts().sort_index().to_dict()}")
    groups = {f"row:{r}": vs for r, (vs, _) in vc.EXP1_ROWS.items()}
    groups.update({f"label:{v}": [v] for v in vlabels})
    out = {}
    for name, vs in groups.items():
        per_rad = lab[vs].max(axis=1).groupby(lab["image_id"]).sum()  # radiologists calling it positive
        out[name] = {"vindr_train_n": len(per_rad),
                     "vindr_train_prev_majority": float((per_rad * 2 > n_rads).mean()),
                     "vindr_train_prev_any": float((per_rad >= 1).mean())}
    return out


def mimic_train(labels_path, test_manifest):
    lab = pd.read_csv(labels_path).drop_duplicates("study_id")
    split_path = os.path.join(os.path.dirname(labels_path), "mimic-cxr-2.0.0-split.csv.gz")
    if os.path.exists(split_path):
        sp = pd.read_csv(split_path)
        train = set(sp.loc[sp["split"] == "train", "study_id"])
        how = f"official train split ({split_path})"
    else:
        test = set(pd.read_csv(test_manifest)["study_id"])
        train = set(lab["study_id"]) - test
        how = "all studies minus our MIMIC test studies (split file not found)"
    lab = lab[lab["study_id"].isin(train)]
    print(f"MIMIC train: {len(lab)} studies, {how}")
    out = {}
    for cx in sorted({c for _, c in vc.EXP1_ROWS.values()}):
        out[cx] = {"mimic_train_n": len(lab), "mimic_train_prev": float((lab[cx] == 1).mean()),
                   "mimic_train_prev_incl_uncertain": float(lab[cx].isin([1, -1]).mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vindr_train", default=vc.VINDR_LABELS_TRAIN)
    ap.add_argument("--mimic_labels", default=MIMIC_LABELS)
    ap.add_argument("--mimic_manifest", default=os.path.join(REPO, "manifests", "mimic.csv"))
    ap.add_argument("--out", default=RESULTS_SCORES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    vd = vindr_train(args.vindr_train)
    mm = mimic_train(args.mimic_labels, args.mimic_manifest)
    rows = []
    for name, v in vd.items():
        kind, finding = name.split(":", 1)
        cx = vc.EXP1_ROWS[finding][1] if kind == "row" else None
        rows.append({"kind": kind, "finding": finding, "chexpert_class": cx, **v, **(mm.get(cx, {}) if cx else {})})
    df = pd.DataFrame(rows)
    path = os.path.join(args.out, "vindr_training_prevalence.csv")
    df.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\nExp 1 rows:")
        print(df[df["kind"] == "row"].drop(columns="kind").to_string(index=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
