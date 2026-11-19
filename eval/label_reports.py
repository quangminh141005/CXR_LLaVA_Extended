"""Step 6.1-6.2: turn generated reports into CheXpert-Labeler labels (each dataset separately).

Two sub-commands (cxreval env), with the labeler run in between (chexpert-label env):

  prepare  results/<ds>.jsonl (last line per key) -> results/labeler/<ds>_reports.csv
           (headerless, 1 report per row, manifest order) + <ds>_keys.csv.
           Stops if a manifest image has no report.
  collect  results/labeler/<ds>_labeled.csv + keys -> results/<ds>_pred_labels.csv
           (key + 14 raw labels: 1 / 0 / -1 / blank)

Usage (on the server, repo root):
    python -u eval/label_reports.py prepare
    # env chexpert-label, in the labeler folder (exact commands printed by prepare):
    python label.py --reports_path <repo>/results/labeler/chexpert_reports.csv --output_path <repo>/results/labeler/chexpert_labeled.csv
    python label.py --reports_path <repo>/results/labeler/mimic_reports.csv    --output_path <repo>/results/labeler/mimic_labeled.csv
    python -u eval/label_reports.py collect
"""
import argparse
import csv
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from validate_labeler import DEFAULT_LABELER_DIR, LABELS  # noqa: E402

DATASETS = ["chexpert", "mimic"]


def load_reports(path):
    """Last record per key (run_inference appends retries)."""
    last = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                last[r["key"]] = r
    return last


def one_line(text):
    return " ".join(str(text).split())


def cmd_prepare(args):
    lab_dir = os.path.join(args.results, "labeler")
    os.makedirs(lab_dir, exist_ok=True)
    cmds, failed = [], False
    for ds in args.datasets:
        man = pd.read_csv(os.path.join(args.manifests, f"{ds}.csv"))
        recs = load_reports(os.path.join(args.results, f"{ds}.jsonl"))
        missing = [k for k in man["key"] if k not in recs or recs[k].get("error") or not recs[k].get("report")]
        extra = set(recs) - set(man["key"])
        print(f"{ds:9s} manifest {len(man)}  reports {len(recs)}  missing/failed {len(missing)}  not in manifest {len(extra)}")
        if missing:
            failed = True
            print("  !! no usable report for: " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
            print("     -> rerun eval/run_inference.py first (it retries failed images)")
            continue
        texts = [one_line(recs[k]["report"]) for k in man["key"]]
        rep_path = os.path.join(lab_dir, f"{ds}_reports.csv")
        with open(rep_path, "w", newline="") as f:
            w = csv.writer(f)
            for t in texts:
                w.writerow([t])
        man[["key"]].to_csv(os.path.join(lab_dir, f"{ds}_keys.csv"), index=False)
        n_tok = pd.Series([recs[k].get("n_tokens", 0) for k in man["key"]])
        print(f"  -> {rep_path}  ({len(texts)} reports, tokens mean {n_tok.mean():.0f}, min {n_tok.min()}, max {n_tok.max()})")
        cmds.append(f"python label.py --reports_path {rep_path} --output_path {os.path.join(lab_dir, ds + '_labeled.csv')}")
    if cmds:
        print(f"\nnext, in the labeler env:\n  conda activate chexpert-label\n  cd {args.labeler_dir}\n  "
              + "\n  ".join(cmds))
    sys.exit(1 if failed else 0)


def cmd_collect(args):
    lab_dir = os.path.join(args.results, "labeler")
    for ds in args.datasets:
        keys = pd.read_csv(os.path.join(lab_dir, f"{ds}_keys.csv"))
        lab = pd.read_csv(os.path.join(lab_dir, f"{ds}_labeled.csv"))
        assert len(lab) == len(keys), f"{ds}: labeled {len(lab)} rows vs keys {len(keys)}"
        out = pd.concat([keys.reset_index(drop=True), lab[LABELS].reset_index(drop=True)], axis=1)
        path = os.path.join(args.results, f"{ds}_pred_labels.csv")
        out.to_csv(path, index=False)
        print(f"{ds:9s} {len(out)} labeled -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--manifests", default=os.path.join(REPO, "manifests"))
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--labeler_dir", default=DEFAULT_LABELER_DIR, help="only used to print the next command")
    sub.add_parser("collect")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
