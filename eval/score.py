"""Step 6.3-6.5: score the labeled reports against ground truth (paper protocol), per dataset.

Reference: Lee et al., European Radiology (2025) 35:4374-4386 (paper/CXR-LLAVA.pdf), Tables 2-3.

GT:          manifests/<ds>.csv (CheXpert: radiologist 0/1; MIMIC: official CheXpert-Labeler labels)
Prediction:  results/labels/<ds>_pred_labels.csv (CheXpert-Labeler on the generated reports)

Per finding, only images where GT AND prediction are definite (1 or 0) are scored
("uncertain" or not mentioned = excluded). Positive class = 1.
  accuracy    (TP+TN)/n        sensitivity TP/(TP+FN)
  specificity TN/(TN+FP)       F1          2TP/(2TP+FP+FN)      (+ precision, not in the paper)
Average rows as in the paper (Average for 4 / 5 pathologies, Overall average):
  micro = pooled TP/FP/FN/TN over the findings  (matches the paper's numbers, Q9)
  macro = mean of the per-finding values        (for reference)
95% CI: non-parametric bootstrap, 1000 resamples of images (seed 0), percentile 2.5 / 97.5;
each resample recomputes every metric incl. the exclusion, so averages get CIs too.

Usage (on the server, repo root, env cxreval):
    python -u eval/score.py 2>&1 | tee log/scoring/score.log
    python -u eval/score.py --datasets chexpert --tag _json_gemini 2>&1 | tee log/scoring/score_json_gemini.log
Output: results/scores/scores_<ds><tag>.csv (long format: one row per table row x metric)
--tag selects results/labels/<ds>_pred_labels<tag>.csv (e.g. _json_gemini = label-prompt route)
    and names the output; --gt replaces the GT manifest (one dataset at a time), e.g.
    python -u eval/score.py --datasets mimic --gt manifests/mimic_gt_full.csv --tag _gt_full
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from validate_labeler import LABELS  # noqa: E402
from paths import RESULTS_LABELS, RESULTS_SCORES  # noqa: E402

FOUR = ["Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
# table rows in paper order: a finding name, or (average name, findings pooled)
ROWS = {
    "mimic": FOUR + [("Average (4)", FOUR), "Pneumonia", "Pneumothorax",
                     ("Overall (6)", FOUR + ["Pneumonia", "Pneumothorax"])],
    "chexpert": FOUR + [("Average (4)", FOUR), "Atelectasis", ("Average (5)", FOUR + ["Atelectasis"]),
                        "Lung Opacity", "Support Devices",
                        ("Overall (7)", FOUR + ["Atelectasis", "Lung Opacity", "Support Devices"])],
}
METRICS = ["accuracy", "sensitivity", "specificity", "F1", "precision"]
PAPER_METRICS = ["accuracy", "sensitivity", "specificity", "F1"]

# paper values (value, CI low, CI high), CXR-LLaVA column of Tables 2 and 3
PAPER = {
    "mimic": {
        "accuracy": {"Cardiomegaly": (.79, .77, .82), "Consolidation": (.93, .91, .96), "Edema": (.81, .77, .85),
                     "Pleural Effusion": (.87, .85, .88), "Average (4)": (.85, .84, .86),
                     "Pneumonia": (.69, .62, .76), "Pneumothorax": (.89, .88, .91), "Overall (6)": (.86, .85, .87)},
        "sensitivity": {"Cardiomegaly": (.88, .85, .90), "Consolidation": (.62, .48, .75), "Edema": (.85, .81, .89),
                        "Pleural Effusion": (.85, .82, .87), "Average (4)": (.85, .84, .87),
                        "Pneumonia": (.53, .42, .64), "Pneumothorax": (.35, .28, .42), "Overall (6)": (.80, .78, .82)},
        "specificity": {"Cardiomegaly": (.55, .49, .61), "Consolidation": (.97, .96, .99), "Edema": (.75, .68, .81),
                        "Pleural Effusion": (.88, .86, .90), "Average (4)": (.85, .83, .86),
                        "Pneumonia": (.87, .78, .95), "Pneumothorax": (.97, .96, .98), "Overall (6)": (.89, .88, .90)},
        "F1": {"Cardiomegaly": (.86, .85, .88), "Consolidation": (.68, .57, .78), "Edema": (.84, .81, .87),
               "Pleural Effusion": (.83, .81, .85), "Average (4)": (.84, .83, .85),
               "Pneumonia": (.65, .54, .74), "Pneumothorax": (.46, .37, .53), "Overall (6)": (.81, .80, .82)},
    },
    "chexpert": {
        "accuracy": {"Cardiomegaly": (.66, .62, .70), "Consolidation": (.67, .64, .71), "Edema": (.72, .68, .76),
                     "Pleural Effusion": (.77, .74, .81), "Average (4)": (.71, .69, .73),
                     "Atelectasis": (.77, .73, .80), "Average (5)": (.72, .70, .74),
                     "Lung Opacity": (.82, .79, .86), "Support Devices": (.76, .72, .79), "Overall (7)": (.74, .73, .75)},
        "sensitivity": {"Cardiomegaly": (.90, .86, .95), "Consolidation": (.93, .83, 1.0), "Edema": (.92, .87, .98),
                        "Pleural Effusion": (.96, .92, .99), "Average (4)": (.93, .90, .95),
                        "Atelectasis": (.85, .79, .91), "Average (5)": (.90, .88, .93),
                        "Lung Opacity": (.90, .87, .93), "Support Devices": (.83, .79, .88),
                        "Overall (7)": (.89, .87, .90)},
        "specificity": {"Cardiomegaly": (.56, .51, .61), "Consolidation": (.66, .62, .70), "Edema": (.68, .64, .73),
                        "Pleural Effusion": (.72, .68, .77), "Average (4)": (.66, .64, .68),
                        "Atelectasis": (.73, .68, .77), "Average (5)": (.67, .65, .69),
                        "Lung Opacity": (.74, .68, .79), "Support Devices": (.68, .62, .74),
                        "Overall (7)": (.68, .66, .70)},
        "F1": {"Cardiomegaly": (.62, .56, .67), "Consolidation": (.24, .17, .31), "Edema": (.50, .43, .57),
               "Pleural Effusion": (.63, .57, .69), "Average (4)": (.53, .49, .56),
               "Atelectasis": (.69, .64, .74), "Average (5)": (.57, .54, .59),
               "Lung Opacity": (.84, .81, .87), "Support Devices": (.78, .74, .82), "Overall (7)": (.67, .65, .69)},
    },
}


def indicators(gt, pred):
    """Boolean arrays TP, FP, FN, TN over images; only definite (1/0) GT and prediction count."""
    valid = np.isin(gt, (0, 1)) & np.isin(pred, (0, 1))
    return (valid & (gt == 1) & (pred == 1), valid & (gt == 0) & (pred == 1),
            valid & (gt == 1) & (pred == 0), valid & (gt == 0) & (pred == 0))


def metrics(tp, fp, fn, tn):
    """Works on scalars or arrays of counts."""
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"accuracy": (tp + tn) / (tp + fp + fn + tn), "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp), "F1": 2 * tp / (2 * tp + fp + fn), "precision": tp / (tp + fp)}


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def score_dataset(ds, gt_df, pred_df, n_boot, seed):
    df = gt_df[["key"] + LABELS].merge(pred_df[["key"] + LABELS], on="key", how="inner", suffixes=("_gt", "_pred"))
    assert len(df) == len(gt_df) == len(pred_df), \
        f"{ds}: key mismatch (manifest {len(gt_df)}, predictions {len(pred_df)}, matched {len(df)})"
    idx = np.random.default_rng(seed).integers(0, len(df), size=(n_boot, len(df)))

    cnt, boot = {}, {}  # per finding: point counts (TP, FP, FN, TN) and bootstrap counts (each shape (B,))
    findings = [r for r in ROWS[ds] if isinstance(r, str)]
    for f in findings:
        ind = indicators(df[f"{f}_gt"].to_numpy(float), df[f"{f}_pred"].to_numpy(float))
        cnt[f] = tuple(int(x.sum()) for x in ind)
        boot[f] = tuple(x[idx].sum(1) for x in ind)

    out = []
    for r in ROWS[ds]:
        name, fs = (r, [r]) if isinstance(r, str) else r
        c = tuple(sum(cnt[f][i] for f in fs) for i in range(4))
        b = tuple(sum(boot[f][i] for f in fs) for i in range(4))
        point, bm = metrics(*c), metrics(*b)
        base = {"dataset": ds, "row": name, "is_average": len(fs) > 1, "n": sum(c), "n_pos": c[0] + c[2],
                "TP": c[0], "FP": c[1], "FN": c[2], "TN": c[3]}
        for m in METRICS:
            lo, hi = ci(bm[m])
            rec = {**base, "metric": m, "value": float(point[m]), "ci_low": lo, "ci_high": hi}
            if len(fs) > 1:  # macro = mean of per-finding values (point and bootstrap)
                rec["macro"] = float(np.nanmean([metrics(*cnt[f])[m] for f in fs]))
                mlo, mhi = ci(np.nanmean(np.stack([metrics(*boot[f])[m] for f in fs]), axis=0))
                rec["macro_ci_low"], rec["macro_ci_high"] = mlo, mhi
            p = PAPER[ds].get(m, {}).get(name)
            if p:
                rec["paper"], rec["paper_ci_low"], rec["paper_ci_high"] = p
                rec["diff"] = rec["value"] - p[0]
            out.append(rec)
    return pd.DataFrame(out), df


def fmt(v, lo=None, hi=None):
    if v is None or pd.isna(v):
        return "-"
    if lo is None or pd.isna(lo):
        return f"{v:.2f}"
    return f"{v:.2f} ({lo:.2f}-{hi:.2f})"


def print_tables(ds, out):
    for m in METRICS:
        sub = out[out["metric"] == m]
        t = pd.DataFrame({
            "row": ["  " + r if not a else r for r, a in zip(sub["row"], sub["is_average"])],
            "n": sub["n"], "n_pos": sub["n_pos"],
            "ours (micro)": [fmt(r["value"], r["ci_low"], r["ci_high"]) for _, r in sub.iterrows()],
            "paper": [fmt(r.get("paper"), r.get("paper_ci_low"), r.get("paper_ci_high")) for _, r in sub.iterrows()],
            "diff": [f"{d:+.2f}" if not pd.isna(d) else "-" for d in sub.get("diff", pd.Series([np.nan] * len(sub)))],
            "ours macro": [fmt(r.get("macro"), r.get("macro_ci_low"), r.get("macro_ci_high"))
                           if r["is_average"] else "" for _, r in sub.iterrows()],
        })
        note = " (not reported in the paper)" if m not in PAPER_METRICS else ""
        print(f"\n== {ds}: {m}{note}, 95% bootstrap CI ==")
        print(t.to_string(index=False))


def print_pred_distribution(ds, df):
    rows = []
    for f in [r for r in ROWS[ds] if isinstance(r, str)]:
        p, g = df[f"{f}_pred"], df[f"{f}_gt"]
        rows.append({"finding": f, "pred 1": int((p == 1).sum()), "pred 0": int((p == 0).sum()),
                     "pred -1": int((p == -1).sum()), "pred blank": int(p.isna().sum()),
                     "GT 1/0": int(g.isin([0, 1]).sum()), "scored n": int((g.isin([0, 1]) & p.isin([0, 1])).sum())})
    print(f"\n-- {ds}: predicted label distribution ({len(df)} images; -1/blank are not scored)")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=list(ROWS), default=["chexpert", "mimic"])
    ap.add_argument("--manifests", default=os.path.join(REPO, "manifests"))
    ap.add_argument("--labels", default=RESULTS_LABELS, help="folder with <ds>_pred_labels<tag>.csv")
    ap.add_argument("--out", default=RESULTS_SCORES, help="folder for scores_<ds><tag>.csv")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="prediction file suffix: <ds>_pred_labels<tag>.csv")
    ap.add_argument("--gt", default=None, help="GT manifest to use instead of manifests/<ds>.csv")
    ap.add_argument("--pred_tag", default=None,
                    help="prediction suffix if different from --tag (default: same as --tag, or '' with --gt)")
    args = ap.parse_args()
    if args.gt and len(args.datasets) != 1:
        ap.error("--gt needs exactly one --datasets")
    pred_tag = args.pred_tag if args.pred_tag is not None else ("" if args.gt else args.tag)
    os.makedirs(args.out, exist_ok=True)

    for ds in args.datasets:
        gt_path = args.gt or os.path.join(args.manifests, f"{ds}.csv")
        pred_path = os.path.join(args.labels, f"{ds}_pred_labels{pred_tag}.csv")
        print(f"\nGT: {gt_path}\npredictions: {pred_path}")
        gt = pd.read_csv(gt_path)
        pred = pd.read_csv(pred_path)
        out, df = score_dataset(ds, gt, pred, args.n_boot, args.seed)
        print(f"\n{'=' * 78}\n{ds.upper()}\n{'=' * 78}")
        print_pred_distribution(ds, df)
        print_tables(ds, out)
        path = os.path.join(args.out, f"scores_{ds}{args.tag}.csv")
        out.to_csv(path, index=False)
        print(f"\n-> {path}")
    print("\nAverages: 'ours (micro)' pools TP/FP/FN/TN over the findings, as the paper does (Q9).")


if __name__ == "__main__":
    main()
