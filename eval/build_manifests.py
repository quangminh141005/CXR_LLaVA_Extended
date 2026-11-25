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
    python -u eval/build_manifests.py 2>&1 | tee log/setup/build_manifests.log
    python -u eval/build_manifests.py --only chexpert
    python -u eval/build_manifests.py --only vindr 2>&1 | tee log/setup/build_manifest_vindr.log

  manifests/vindr.csv        (extension) 3000 VinDr test PNGs, width/height, mapped CheXpert GT columns,
                             VinDr labels as vd_<label>; + manifests/vindr_boxes.csv (union box area per
                             image x finding, part kept by the center crop). See eval/vindr_common.py.
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


# ------------------------------------------------------------------ VinDr (extension experiments)
def build_vindr(a):
    """manifests/vindr.csv (one row per test image) + manifests/vindr_boxes.csv (one row per image x finding
    with boxes: union area fraction of the image, fraction of that union kept by the center crop)."""
    from PIL import Image
    import vindr_common as vc

    header("VinDr-CXR test")
    lab = pd.read_csv(vc.VINDR_LABELS_TEST)
    vlabels = [c for c in lab.columns if c not in vc.NON_LABEL_COLS]
    print(f"labels: {len(lab)} rows, {lab['image_id'].nunique()} images, rows per image "
          f"{dict(Counter(lab.groupby('image_id').size()))}  ({vc.VINDR_LABELS_TEST})")
    print(f"VinDr label columns ({len(vlabels)}): {vlabels}")
    needed = {v for vs, _ in vc.EXP1_ROWS.values() for v in vs} | set(vc.YESNO_FINDINGS) | {"No finding"}
    unknown = sorted(needed - set(vlabels))
    assert not unknown, f"names in vindr_common.py not in the label file: {unknown}"
    vals = sorted({str(v) for c in vlabels for v in lab[c].unique()})
    print(f"label values present: {vals}  (expected 0/1)")
    img = lab.groupby("image_id")[vlabels].max().reset_index()  # test = consensus; max() only if >1 row

    rows, problems = [], []
    for i in img["image_id"]:
        p = os.path.join(vc.VINDR_PNG_DIR, f"{i}.png")
        try:
            with Image.open(p) as im:  # header only
                w, h = im.size
        except OSError as e:
            problems.append((p, f"unreadable: {e}"))
            w = h = None
        rows.append({"key": f"VinDr:{i}", "dataset": "VinDr", "image_id": i, "image_path": p, "study_id": i,
                     "width": w, "height": h})
    df = pd.DataFrame(rows)
    if problems:
        return df, problems
    for cx in LABELS:  # mapped CheXpert GT (blank where VinDr has no equivalent, e.g. Support Devices)
        src = vc.CHEXPERT_FROM_VINDR.get(cx)
        df[cx] = img[src].max(axis=1).to_numpy() if src else float("nan")
    for v in vlabels:
        df[vc.vd(v)] = img[v].to_numpy()

    print("\npositives per VinDr label (test, consensus):")
    print(", ".join(f"{v} {int(img[v].sum())}" for v in vlabels))

    # boxes
    bx = pd.read_csv(vc.VINDR_BOXES_TEST)
    bx = bx[bx["class_name"] != "No finding"].dropna(subset=["x_min", "y_min", "x_max", "y_max"])
    rads = f", rad_ids {sorted(bx['rad_id'].unique())[:10]}" if "rad_id" in bx.columns else \
        " (no rad_id column: consensus boxes)"
    print(f"\nboxes file columns: {list(bx.columns)}")
    print(f"boxes: {len(bx)} (excl. 'No finding'), images with boxes {bx['image_id'].nunique()}{rads}")
    size = df.set_index("image_id")[["width", "height"]]
    bx = bx.join(size, on="image_id")
    outside = bx[(bx["x_max"] > bx["width"] + 1) | (bx["y_max"] > bx["height"] + 1) |
                 (bx["x_min"] < -1) | (bx["y_min"] < -1)]
    print(f"boxes outside the PNG: {len(outside)}  (0 expected: PNG = DICOM pixel grid)")
    if len(outside):
        problems.append(("(boxes)", f"{len(outside)} boxes outside the PNG, e.g. {outside.iloc[0].to_dict()}"))
    lab_no_box = sorted(set(vlabels) - set(bx["class_name"]))
    print(f"labels without any box (global labels): {lab_no_box}")

    # one row per image x group; group = an Exp 1 row (its VinDr labels' boxes merged, e.g. rib + clavicle
    # fracture) or a single VinDr label. Union over all boxes of the group (all radiologists listed).
    groups = [("row", r, vs) for r, (vs, _) in vc.EXP1_ROWS.items()] + \
             [("label", c, [c]) for c in sorted(bx["class_name"].unique())]
    brows = []
    for kind, gname, classes in groups:
        sub = bx[bx["class_name"].isin(classes)]
        for iid, g in sub.groupby("image_id"):
            w, h = size.loc[iid]
            boxes = list(g[["x_min", "y_min", "x_max", "y_max"]].itertuples(index=False, name=None))
            u = vc.union_area(boxes)
            kept = vc.union_area(vc.clip(boxes, vc.crop_box(w, h)))
            brows.append({"kind": kind, "group": gname, "image_id": iid, "n_boxes": len(boxes), "union_area": u,
                          "area_frac": u / (w * h), "kept_frac": kept / u if u else float("nan")})
    bdf = pd.DataFrame(brows)
    lab_rows = bdf[bdf["kind"] == "label"]
    print(f"image x VinDr label with boxes: {len(lab_rows)}; area_frac quantiles 10/50/90% "
          f"{lab_rows['area_frac'].quantile([.1, .5, .9]).round(4).tolist()}; "
          f"union kept by the center crop < 50%: {int((lab_rows['kept_frac'] < .5).sum())}")

    if problems:  # don't write a half-checked manifest
        return df, problems
    bpath = os.path.join(a.out, "vindr_boxes.csv")
    bdf.to_csv(bpath, index=False)
    print(f"-> {bpath}  ({len(bdf)} rows)")
    return df, []


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "manifests"))
    ap.add_argument("--only", choices=["chexpert", "mimic", "vindr"], nargs="*",
                    help="default: chexpert + mimic; vindr is built only when named here")
    ap.add_argument("--chexpert_root", default=DEFAULTS["chexpert_root"])
    ap.add_argument("--mimic_csv", default=DEFAULTS["mimic_csv"])
    ap.add_argument("--mimic_jpg_root", default=MIMIC_JPG_ROOT)
    ap.add_argument("--mimic_report_root", default=DEFAULTS["mimic_report_root"])
    ap.add_argument("--mimic_labels", default=MIMIC_LABELS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    failed = False
    for name, fn in (("chexpert", build_chexpert), ("mimic", build_mimic), ("vindr", build_vindr)):
        if (a.only and name not in a.only) or (not a.only and name == "vindr"):
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
