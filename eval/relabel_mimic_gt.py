"""Relabel the MIMIC ground truth ourselves, as the paper did (instead of the official labels).

Why: the official mimic-cxr-2.0.0-chexpert.csv labels only ONE section of each original
report (impression > findings > ...). The paper ran CheXpert-Labeler on the reports itself
(supplement, "Label Distribution in the MIMIC Internal Test Set") and has far more definite
labels (e.g. Pneumothorax 1665 neg / 222 pos vs official 561 / 149). The predictions
(results/mimic_pred_labels.csv) are already labeled from the FULL generated report and are
not touched here. Which text the authors labeled is not stated, so two variants are tested:

  full     the whole original report (whitespace collapsed)
  findimp  FINDINGS + IMPRESSION sections only (MIMIC-CXR section parser); falls back to
           the full report when neither section exists (counted)

Sub-commands (cxreval env), with the labeler run in between (chexpert-label env):
  prepare  manifests/mimic.csv -> results/mimic_gt_relabel/<variant>_reports.csv (+ keys.csv)
  compare  labeler outputs -> label counts per variant vs official vs paper supplement
  build    --variant V -> manifests/mimic_gt_<V>.csv (manifest copy, labels replaced)

Usage (on the server, repo root):
    python -u eval/relabel_mimic_gt.py prepare
    # env chexpert-label, in the labeler folder (exact commands printed by prepare)
    python -u eval/relabel_mimic_gt.py compare 2>&1 | tee log/mimic_gt_compare.log
    python -u eval/relabel_mimic_gt.py build --variant full
    python -u eval/score.py --datasets mimic --gt manifests/mimic_gt_full.csv --tag _gt_full \\
        2>&1 | tee log/score_mimic_gt_full.log

Outputs contain MIMIC report text / labels (PhysioNet DUA): results/ and manifests/ are gitignored.
"""
import argparse
import csv
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
import mimic_section_parser as sp  # noqa: E402
from validate_labeler import DEFAULT_LABELER_DIR, LABELS, MIMIC_FINDINGS, list_rindex  # noqa: E402

VARIANTS = ["full", "findimp"]
# supplement, "Label Distribution in the MIMIC Internal Test Set": (negative, positive, uncertain/missing)
PAPER_COUNTS = {
    "Enlarged Cardiomediastinum": (894, 238, 1868), "Cardiomegaly": (528, 950, 1522),
    "Lung Lesion": (20, 166, 2814), "Lung Opacity": (71, 1018, 1911), "Edema": (473, 541, 1986),
    "Consolidation": (689, 226, 2085), "Pneumonia": (227, 471, 2302), "Atelectasis": (29, 828, 2143),
    "Pneumothorax": (1665, 222, 1113), "Pleural Effusion": (1362, 859, 779),
    "Pleural Other": (3, 62, 2935), "Fracture": (42, 162, 2796), "Support Devices": (46, 1268, 1686),
}


def one_line(text):
    return " ".join(str(text).split())


def findimp(text):
    """FINDINGS + IMPRESSION (last occurrence of each, as the official extraction) or None."""
    sections, names, _ = sp.section_text(text)
    parts = [sections[list_rindex(names, n)].strip() for n in ("findings", "impression") if n in names]
    parts = [p for p in parts if p]
    return " ".join(parts) if parts else None


def out_dir(args):
    return os.path.join(args.results, "mimic_gt_relabel")


def cmd_prepare(args):
    man = pd.read_csv(os.path.join(args.manifests, "mimic.csv"))
    texts = {v: [] for v in VARIANTS}
    fallback, unreadable = [], []
    for _, r in man.iterrows():
        try:
            with open(r["report_path"]) as f:
                raw = f.read()
        except OSError as e:
            unreadable.append(f"{r['study_id']}: {e}")
            continue
        fi = findimp(raw)
        if fi is None:
            fallback.append(r["study_id"])
        texts["full"].append(one_line(raw))
        texts["findimp"].append(one_line(fi if fi is not None else raw))
    if unreadable:
        print(f"!! {len(unreadable)} unreadable reports, nothing written:\n  " + "\n  ".join(unreadable[:10]))
        sys.exit(1)

    d = out_dir(args)
    os.makedirs(d, exist_ok=True)
    man[["key", "study_id"]].to_csv(os.path.join(d, "keys.csv"), index=False)
    cmds = []
    for v in VARIANTS:
        p = os.path.join(d, f"{v}_reports.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            for t in texts[v]:
                w.writerow([t])
        n_words = pd.Series([len(t.split()) for t in texts[v]])
        print(f"{v:8s} {len(texts[v])} reports -> {p}  (words mean {n_words.mean():.0f}, max {n_words.max()})")
        cmds.append(f"python label.py --reports_path {p} --output_path {os.path.join(d, v + '_labeled.csv')}")
    print(f"findimp: no FINDINGS/IMPRESSION section in {len(fallback)} reports -> full text used")
    print(f"\nnext, in the labeler env:\n  conda activate chexpert-label\n  cd {args.labeler_dir}\n  "
          + "\n  ".join(cmds))


def load_variant(args, v):
    d = out_dir(args)
    keys = pd.read_csv(os.path.join(d, "keys.csv"))
    lab = pd.read_csv(os.path.join(d, f"{v}_labeled.csv"))
    assert len(lab) == len(keys), f"{v}: labeled {len(lab)} rows vs keys {len(keys)}"
    return pd.concat([keys.reset_index(drop=True), lab[LABELS].reset_index(drop=True)], axis=1)


def counts(df, f):
    v = df[f]
    return int((v == 0).sum()), int((v == 1).sum()), int((~v.isin([0, 1])).sum())


def cmd_compare(args):
    sources = {"official": pd.read_csv(os.path.join(args.manifests, "mimic.csv"))}
    for v in VARIANTS:
        try:
            sources[v] = load_variant(args, v)
        except FileNotFoundError as e:
            print(f"!! {v}: {e} (labeler not run?) -> skipped")
    n = len(sources["official"])
    print(f"MIMIC ground truth label counts, {n} studies: neg / pos / uncertain+missing")
    rows, err = [], {s: 0 for s in sources}
    err6 = dict(err)
    for f, paper in PAPER_COUNTS.items():
        row = {"finding": f + (" *" if f in MIMIC_FINDINGS else ""), "paper": "/".join(map(str, paper))}
        for s, df in sources.items():
            c = counts(df, f)
            row[s] = "/".join(map(str, c))
            e = sum(abs(a - b) for a, b in zip(c, paper)) // 2  # = labels that would have to move
            row[f"{s} diff"] = e
            err[s] += e
            if f in MIMIC_FINDINGS:
                err6[s] += e
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n'diff' = labels that differ in class from the paper's counts ((|dneg|+|dpos|+|dunc|)/2);"
          " * = the 6 scored findings")
    print(f"\n{'source':9s} {'13 findings':>12s} {'6 scored':>9s}   (total diff vs paper, lower = closer;"
          f" max {n} per finding)")
    for s in sources:
        print(f"{s:9s} {err[s]:12d} {err6[s]:9d}")
    best = min((s for s in sources if s != "official"), key=lambda s: err6[s], default=None)
    if best:
        print(f"\nclosest relabeling on the 6 scored findings: {best}"
              f"  -> python -u eval/relabel_mimic_gt.py build --variant {best}")


def cmd_build(args):
    man = pd.read_csv(os.path.join(args.manifests, "mimic.csv"))
    lab = load_variant(args, args.variant)
    assert (lab["key"].values == man["key"].values).all(), "key order mismatch with manifests/mimic.csv"
    out = man.copy()
    out[LABELS] = lab[LABELS].values
    path = os.path.join(args.manifests, f"mimic_gt_{args.variant}.csv")
    out.to_csv(path, index=False)
    print(f"-> {path}  ({len(out)} rows; labels from '{args.variant}' relabeling, rest = manifests/mimic.csv)")
    print(f"score: python -u eval/score.py --datasets mimic --gt {path} --tag _gt_{args.variant}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", default=os.path.join(REPO, "manifests"))
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--labeler_dir", default=DEFAULT_LABELER_DIR, help="only used to print the next command")
    sub.add_parser("compare")
    p = sub.add_parser("build")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "compare": cmd_compare, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
