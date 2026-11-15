"""Step 0: read-only data check for the CXR-LLaVA reproduction.

Checks that the test images / reports / labels of the 4 evaluation sets exist and
can be read. Nothing is modified; results are printed and written to --out.

Usage (on the server):
    python eval/check_data.py --out data_check
    python eval/check_data.py --out data_check --only mimic --verify

Status per file:
    ok          file exists and its first bytes can be read
    missing     file does not exist
    unreadable  path exists (or its folder does) but reading fails, e.g. a broken
                network mount ("Transport endpoint is not connected")
"""
import argparse
import errno
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = dict(
    mimic_csv=os.path.join(REPO, "MIMIC_CXR_DATASET_TESTSET.csv"),
    mimic_jpg_roots=[
        "/storage/shared/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
        "/storage/shared/MIMIC-CXR-JPG/split-4/files",
        "/storage/shared/MIMIC-CXR-JPG/split-4",
    ],
    mimic_report_root="/storage/shared/MIMIC-CXR-Report/download-mimic-cxr-txt/files",
    mimic_label_search_root="/storage/shared/MIMIC-CXR-JPG",
    chexpert_root="/storage/student4/hanhpm/datasets/CheXpert_test_labels",
    vindr_png_dir="/storage/shared/VinDr-CXR/processed_images/png",
    vindr_labels="/storage/shared/VinDr-CXR/annotations/image_labels_test.csv",
    indiana_root="/storage/student4/nqminh/CLEAR-experiments/data/raw/indiana",
)

FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Pleural Effusion", "Pneumothorax"]


# ---------------------------------------------------------------- helpers
def file_status(path, verify=False):
    try:
        with open(path, "rb") as f:
            f.read(16)
    except FileNotFoundError:
        return "missing"
    except OSError as e:
        if e.errno == errno.ENOENT:
            return "missing"
        return "unreadable"
    if verify:
        try:
            from PIL import Image
            with Image.open(path) as im:
                im.verify()
        except Exception:
            return "unreadable"
    return "ok"


def statuses(paths, verify=False, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: file_status(p, verify), paths))


def index_files(root, exts):
    """Map basename -> full path for all files under root with given extensions."""
    idx, errors = {}, 0
    def onerror(_):
        nonlocal errors
        errors += 1
    for dirpath, _, files in os.walk(root, onerror=onerror):
        for fn in files:
            if fn.lower().endswith(exts):
                idx.setdefault(fn, os.path.join(dirpath, fn))
    return idx, errors


def write_list(out, name, items):
    if not items:
        return
    path = os.path.join(out, name)
    with open(path, "w") as f:
        f.write("\n".join(items) + "\n")
    print(f"    -> list written: {path}")


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------- MIMIC
def find_mimic_label_csv(root, max_depth=6):
    """Search for mimic-cxr-2.0.0-chexpert.csv(.gz) without walking image folders."""
    hits = []
    root_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, files in os.walk(root, onerror=lambda e: None):
        # prune patient image folders (p10..p19, pXXXXXXXX) and .git
        dirnames[:] = [d for d in dirnames
                       if not re.fullmatch(r"p\d+", d) and d != ".git"]
        if dirpath.count("/") - root_depth >= max_depth:
            dirnames[:] = []
        for fn in files:
            if "chexpert" in fn.lower() and fn.lower().endswith((".csv", ".csv.gz")):
                hits.append(os.path.join(dirpath, fn))
    return hits


def check_mimic(a, out):
    header("MIMIC-CXR (JPG)")
    rows = pd.read_csv(a.mimic_csv, header=None)[0].tolist()
    rels = []
    for r in rows:
        r = r.strip().replace("\\", "/")
        if r.startswith("files/"):
            r = r[len("files/"):]
        rels.append(re.sub(r"\.dcm$", ".jpg", r))
    print(f"test list: {len(rels)} images ({a.mimic_csv})")

    per_root = {}
    for root in a.mimic_jpg_roots:
        st = statuses([os.path.join(root, r) for r in rels], a.verify)
        per_root[root] = st
        c = Counter(st)
        print(f"  {root}\n    ok={c['ok']}  missing={c['missing']}  unreadable={c['unreadable']}")

    chosen, best = [], []
    for i, r in enumerate(rels):
        pick = next((root for root in a.mimic_jpg_roots if per_root[root][i] == "ok"), None)
        chosen.append(os.path.join(pick, r) if pick else "")
        best.append("ok" if pick else
                    ("unreadable" if any(per_root[root][i] == "unreadable" for root in a.mimic_jpg_roots)
                     else "missing"))
    c = Counter(best)
    print(f"  COMBINED (any copy ok): ok={c['ok']}  missing={c['missing']}  unreadable={c['unreadable']}")

    # reports: pXX/pXXXXXXXX/sXXXXXXXX.txt
    rep_paths = [os.path.join(a.mimic_report_root, os.path.dirname(r) + ".txt") for r in rels]
    rst = statuses(rep_paths)
    rc = Counter(rst)
    print(f"reports ({a.mimic_report_root}): ok={rc['ok']}  missing={rc['missing']}  unreadable={rc['unreadable']}")

    both = sum(1 for s1, s2 in zip(best, rst) if s1 == "ok" and s2 == "ok")
    print(f"usable (image ok AND report ok): {both}/{len(rels)}")

    pd.DataFrame({"rel": rels, "image_path": chosen, "image_status": best,
                  "report_path": rep_paths, "report_status": rst}) \
        .to_csv(os.path.join(out, "mimic_status.csv"), index=False)
    print(f"    -> per-image table: {os.path.join(out, 'mimic_status.csv')}")
    write_list(out, "mimic_images_not_ok.txt", [r for r, s in zip(rels, best) if s != "ok"])
    write_list(out, "mimic_reports_not_ok.txt", [p for p, s in zip(rep_paths, rst) if s != "ok"])

    print(f"searching label CSV under {a.mimic_label_search_root} ...")
    hits = find_mimic_label_csv(a.mimic_label_search_root)
    if not hits:
        print("  label CSV NOT found -> fallback: CheXbert on MIMIC reference reports")
        return
    for h in hits:
        print(f"  found: {h}")
    try:
        lab = pd.read_csv(hits[0])
        test_studies = {int(os.path.basename(os.path.dirname(r))[1:]) for r in rels}
        covered = test_studies & set(lab["study_id"].astype(int))
        print(f"  {hits[0]}: {len(lab)} rows, columns={list(lab.columns)}")
        print(f"  test studies covered by labels: {len(covered)}/{len(test_studies)}")
    except Exception as e:
        print(f"  could not read label CSV: {e!r}")


# ---------------------------------------------------------------- CheXpert
def check_chexpert(a, out):
    header("CheXpert test")
    gt_path = os.path.join(a.chexpert_root, "groundtruth.csv")
    gt = pd.read_csv(gt_path)
    print(f"groundtruth: {len(gt)} studies ({gt_path})")
    missing_cols = [f for f in FINDINGS if f not in gt.columns]
    print(f"6 findings present in columns: {not missing_cols}  {missing_cols or ''}")

    idx, errs = index_files(a.chexpert_root, (".jpg",))
    frontal = {}
    # index by "patientX/studyY" -> frontal jpgs
    for dirpath, _, files in os.walk(a.chexpert_root, onerror=lambda e: None):
        m = re.search(r"(patient\d+/study\d+)$", dirpath)
        if m:
            fr = sorted(f for f in files if f.endswith("_frontal.jpg"))
            if fr:
                frontal[m.group(1)] = os.path.join(dirpath, fr[0])
    keys = [re.search(r"(patient\d+/study\d+)", s).group(1) for s in gt["Study"]]
    found = [k for k in keys if k in frontal]
    st = statuses([frontal[k] for k in found], a.verify)
    print(f"jpg files under root: {len(idx)} (walk errors: {errs})")
    print(f"studies with a frontal image: {len(found)}/{len(keys)}  readable: {Counter(st)['ok']}")
    if found:
        print(f"example path: {frontal[found[0]]}")
    write_list(out, "chexpert_no_frontal.txt", [k for k in keys if k not in frontal])
    pos = {f: int((gt[f] == 1).sum()) for f in FINDINGS if f in gt.columns}
    print(f"positives per finding: {pos}")


# ---------------------------------------------------------------- VinDr
def check_vindr(a, out):
    header("VinDr-CXR test")
    lab = pd.read_csv(a.vindr_labels)
    ids = lab["image_id"].drop_duplicates().tolist()
    print(f"labels: {len(lab)} rows, {len(ids)} unique images ({a.vindr_labels})")
    vcols = {f: next((c for c in lab.columns if c.lower() == f.lower()), None) for f in FINDINGS}
    print(f"finding columns: {vcols}")
    paths = [os.path.join(a.vindr_png_dir, f"{i}.png") for i in ids]
    st = statuses(paths, a.verify)
    c = Counter(st)
    print(f"png ({a.vindr_png_dir}): ok={c['ok']}  missing={c['missing']}  unreadable={c['unreadable']}")
    write_list(out, "vindr_not_ok.txt", [p for p, s in zip(paths, st) if s != "ok"])
    pos = {f: int((lab.drop_duplicates("image_id")[col] == 1).sum()) for f, col in vcols.items() if col}
    print(f"positives per finding: {pos}")


# ---------------------------------------------------------------- Indiana / NLMCXR
def check_indiana(a, out):
    header("NLMCXR / Indiana (Kaggle version)")
    rep = pd.read_csv(os.path.join(a.indiana_root, "indiana_reports.csv"))
    proj = pd.read_csv(os.path.join(a.indiana_root, "indiana_projections.csv"))
    print(f"reports: {len(rep)} rows, columns={list(rep.columns)}")
    print(f"projections: {len(proj)} rows, columns={list(proj.columns)}")
    print(f"projection values: {dict(Counter(proj['projection']))}")

    def nonempty(s):
        return s.fillna("").astype(str).str.strip().str.len() > 0
    rep["has_text"] = nonempty(rep["findings"]) | nonempty(rep["impression"])
    print(f"reports with non-empty findings or impression: {int(rep['has_text'].sum())}")

    fr = proj[proj["projection"].str.lower() == "frontal"].sort_values("filename")
    first_frontal = fr.groupby("uid")["filename"].first()
    cand = rep[rep["has_text"] & rep["uid"].isin(first_frontal.index)]
    print(f"studies with text AND >=1 frontal image: {len(cand)}")

    idx, errs = index_files(os.path.join(a.indiana_root, "images"), (".png", ".jpg"))
    print(f"image files indexed: {len(idx)} (walk errors: {errs})")
    files = [first_frontal[u] for u in cand["uid"]]
    found = [idx.get(f) for f in files]
    st = statuses([p for p in found if p], a.verify)
    print(f"candidate frontal images found: {sum(p is not None for p in found)}/{len(files)}  "
          f"readable: {Counter(st)['ok']}")
    if any(found):
        print(f"example path: {next(p for p in found if p)}")
    write_list(out, "indiana_missing.txt", [f for f, p in zip(files, found) if p is None])


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data_check")
    p.add_argument("--only", choices=["mimic", "chexpert", "vindr", "indiana"], nargs="*")
    p.add_argument("--verify", action="store_true", help="also decode each image with PIL (slower)")
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k, default=v, nargs="+" if isinstance(v, list) else None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    checks = dict(chexpert=check_chexpert, vindr=check_vindr,
                  indiana=check_indiana, mimic=check_mimic)
    for name, fn in checks.items():
        if a.only and name not in a.only:
            continue
        try:
            fn(a, a.out)
        except Exception as e:
            print(f"\n!! {name} check failed: {e!r}", file=sys.stderr)
    print("\nDone. Paste everything above back to Claude.")


if __name__ == "__main__":
    main()
