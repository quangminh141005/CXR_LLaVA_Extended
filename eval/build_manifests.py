"""Step 4: build the test-set manifests (one CSV per dataset, kept separate).

A manifest is the fixed list of test images: one row per image with its exact path and
its raw ground truth (14 CheXpert labels: 1 / 0 / -1 / blank). Every later step reads it
(inference: one report per row; labeling/scoring: join on `key`). Images are not copied.

  manifests/chexpert.csv  every *_frontal.jpg of the 500 CheXpert test studies (expect 518),
                          labels per study from groundtruth.csv (radiologist consensus, 0/1)
  manifests/mimic.csv     the 3000 images of MIMIC_CXR_DATASET_TESTSET.csv (split-4 JPGs),
                          labels per study_id from the official mimic-cxr-2.0.0-chexpert.csv,
                          + report_path (original report, not used for scoring)

Read-only on /storage: CheXpert lists one small study folder at a time, MIMIC uses exact
paths from the CSV (no big folder walks). Stops without writing a manifest if any image
is missing or unreadable.

Usage (on the server, repo root, env cxreval):
    python -u eval/build_manifests.py 2>&1 | tee log/build_manifests.log
    python -u eval/build_manifests.py --only chexpert
"""
import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from check_data import DEFAULTS, statuses  # noqa: E402  (same paths / file check as Step 0)
from validate_labeler import LABELS, MIMIC_FINDINGS, MIMIC_LABELS  # noqa: E402

MIMIC_JPG_ROOT = "/storage/shared/MIMIC-CXR-JPG/split-4/files"  # complete copy (Step 0)
CHEXPERT_FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
                     "Lung Opacity", "Support Devices"]
# positives in the paper's CheXpert test set (518 images), for the Q8 check
PAPER_CHEXPERT_POS = {"Atelectasis": 158, "Cardiomegaly": 154, "Consolidation": 29, "Edema": 79,
                      "Pleural Effusion": 105, "Pneumothorax": 9}
COLUMNS = ["key", "dataset", "image_id", "image_path", "study_id"] + LABELS


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def label_counts(df, findings):
    """Per finding: how many 1 / 0 / -1 / blank."""
    rows = []
    for f in findings:
        v = df[f]
        rows.append({"finding": f, "1": int((v == 1).sum()), "0": int((v == 0).sum()),
                     "-1": int((v == -1).sum()), "blank": int(v.isna().sum())})
    return pd.DataFrame(rows).set_index("finding")


def check_files(df):
    """Read 16 bytes of every image (as Step 0). Returns list of (path, status) not ok."""
    st = statuses(df["image_path"].tolist())
    print(f"image files: {dict(Counter(st))}")
    return [(p, s) for p, s in zip(df["image_path"], st) if s != "ok"]


# ------------------------------------------------------------------ CheXpert
def build_chexpert(a):
    header("CheXpert test")
    gt_path = os.path.join(a.chexpert_root, "groundtruth.csv")
    gt = pd.read_csv(gt_path)
    gt["study_id"] = gt["Study"].str.extract(r"(patient\d+/study\d+)")[0]
    bad = gt["study_id"].isna().sum()
    assert bad == 0, f"{bad} rows of groundtruth.csv without patientX/studyY"
    assert gt["study_id"].is_unique, "duplicate studies in groundtruth.csv"
    missing_cols = [c for c in LABELS if c not in gt.columns]
    if missing_cols:
        print(f"!! label columns missing in groundtruth.csv (left blank): {missing_cols}")
        for c in missing_cols:
            gt[c] = float("nan")
    print(f"groundtruth: {len(gt)} studies ({gt_path})")

    img_root = os.path.join(a.chexpert_root, "CheXpert", "test", "test")
    rows, per_study, errors = [], {}, []
    for _, g in gt.iterrows():
        d = os.path.join(img_root, g["study_id"])
        try:
            frontal = sorted(f for f in os.listdir(d) if f.endswith("_frontal.jpg"))
        except OSError as e:
            errors.append((d, str(e)))
            frontal = []
        per_study[g["study_id"]] = len(frontal)
        for f in frontal:
            image_id = f"{g['study_id']}/{f[:-len('.jpg')]}"
            rows.append({"key": f"CheXpert:{image_id}", "dataset": "CheXpert", "image_id": image_id,
                         "image_path": os.path.join(d, f), "study_id": g["study_id"],
                         **{c: g[c] for c in LABELS}})
    df = pd.DataFrame(rows, columns=COLUMNS)

    n_per = Counter(per_study.values())
    print(f"frontal images: {len(df)}  (paper: 518)   studies by number of frontal images: "
          f"{dict(sorted(n_per.items()))}")
    for sid, n in per_study.items():
        if n != 1:
            print(f"  {sid}: {n} frontal")
    vals = sorted({str(v) for c in LABELS for v in df[c].unique()})
    print(f"label values present: {vals}  (expected only 0/1)")

    cnt = label_counts(df, CHEXPERT_FINDINGS + ["Pneumothorax"])
    cnt["paper_pos"] = [PAPER_CHEXPERT_POS.get(f, "") for f in cnt.index]
    print("\nlabel counts per image (scored findings + Pneumothorax for the Q8 check):")
    print(cnt.to_string())

    problems = [(p, f"listdir failed: {e}") for p, e in errors] + check_files(df)
    return df, problems


# ------------------------------------------------------------------ MIMIC
def build_mimic(a):
    header("MIMIC-CXR")
    raw = pd.read_csv(a.mimic_csv, header=None)[0]
    rows = []
    for r in raw:
        rel = r.strip().replace("\\", "/")
        rel = rel[len("files/"):] if rel.startswith("files/") else rel
        rel = re.sub(r"\.dcm$", ".jpg", rel)
        m = re.search(r"(p\d+/p\d+)/s(\d+)/", rel)
        image_id = os.path.splitext(os.path.basename(rel))[0]
        rows.append({"key": f"MIMIC:{image_id}", "dataset": "MIMIC", "image_id": image_id,
                     "image_path": os.path.join(a.mimic_jpg_root, rel), "study_id": int(m.group(2)),
                     "report_path": os.path.join(a.mimic_report_root, m.group(1), f"s{m.group(2)}.txt")})
    df = pd.DataFrame(rows)
    assert df["key"].is_unique, "duplicate images in the MIMIC test list"
    print(f"test list: {len(df)} images, {df['study_id'].nunique()} studies ({a.mimic_csv})")

    lab = pd.read_csv(a.mimic_labels).drop_duplicates("study_id")[["study_id"] + LABELS]
    df = df.merge(lab, on="study_id", how="left", indicator=True)
    no_lab = int((df["_merge"] != "both").sum())
    print(f"studies with official labels: {len(df) - no_lab}/{len(df)}")
    df = df[COLUMNS + ["report_path"]]

    print("\nlabel counts (official CheXpert-Labeler on original reports):")
    print(label_counts(df, MIMIC_FINDINGS).to_string())

    problems = check_files(df)
    if no_lab:
        problems.append(("(labels)", f"{no_lab} studies without official labels"))
    return df, problems


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "manifests"))
    ap.add_argument("--only", choices=["chexpert", "mimic"], nargs="*")
    ap.add_argument("--chexpert_root", default=DEFAULTS["chexpert_root"])
    ap.add_argument("--mimic_csv", default=DEFAULTS["mimic_csv"])
    ap.add_argument("--mimic_jpg_root", default=MIMIC_JPG_ROOT)
    ap.add_argument("--mimic_report_root", default=DEFAULTS["mimic_report_root"])
    ap.add_argument("--mimic_labels", default=MIMIC_LABELS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    failed = False
    for name, fn in (("chexpert", build_chexpert), ("mimic", build_mimic)):
        if a.only and name not in a.only:
            continue
        df, problems = fn(a)
        path = os.path.join(a.out, f"{name}.csv")
        if problems:
            failed = True
            print(f"\n!! {len(problems)} problems -> {path} NOT written:")
            for p, why in problems[:20]:
                print(f"  {why}: {p}")
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more")
            continue
        df.to_csv(path, index=False)
        print(f"\n-> {path}  ({len(df)} rows)")
    print("\nDone." if not failed else "\nDone, with problems (see above).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
