"""Extension experiments on the VinDr-CXR test set: all tables + figures.

Inputs (each part is skipped with a message if its input is missing):
  manifests/vindr.csv, manifests/vindr_boxes.csv           (build_manifests.py --only vindr)
  results/labels/vindr_pred_labels.csv                     (run_inference --datasets vindr + label_reports)
  results/scores/vindr_training_prevalence.csv             (training_prevalence.py)
  results/inference/vindr_yesno.jsonl, vindr_yesno_prior.jsonl   (run_yesno.py real / prior)

Definitions
  GT          VinDr test consensus label = 1 (an Exp 1 row with several VinDr labels = any of them).
  Report hit  CheXpert-Labeler on the generated report: strict = 1 [main]; lenient = 1 or -1 (Q2).
              0 (denied) and blank (omitted) are misses -> unlike score.py, blanks are NOT dropped.
  Specificity (context) = GT negative and the report does not call it positive (strict).
  Size        union of the row's boxes / image area -> <1%, 1-5%, >5%; kept_frac = part of that union
              inside the model's center crop (resize shortest edge + center crop).
  p_yes       P(yes)/(P(yes)+P(no)) at the first answer token; decision threshold 0.5.
  CI          95% percentile bootstrap, B resamples of images (seed 0), as in score.py; deltas between
              prompts are paired (same resamples). AUROC only for findings with >= --min_pos positives.

Outputs: results/scores/vindr_exp1_*.csv, vindr_exp2_*.csv, results/figures/vindr_*.png
Usage (server, repo root, env cxreval):
    python -u eval/analyze_vindr.py 2>&1 | tee log/scoring/analyze_vindr.log
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from paths import RESULTS_FIGS, RESULTS_INF, RESULTS_LABELS, RESULTS_SCORES  # noqa: E402
from validate_labeler import LABELS  # noqa: E402
import vindr_common as vc  # noqa: E402

PROMPTS = ["plain", "guided", "control"]


# ---------------------------------------------------------------- statistics
def ci(b):
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))) if len(b) else (np.nan, np.nan)


def rate(num, den, idx):
    """Ratio of per-image sums, e.g. sensitivity = hits / positives. Returns (point, bootstrap array)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num.sum() / den.sum() if den.sum() else np.nan), num[idx].sum(1) / den[idx].sum(1)


def auroc(s, y):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auroc_boot(s, y, idx):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    return auroc(s, y), np.array([auroc(s[i], y[i]) for i in idx])


def put(rec, name, point, boot):
    lo, hi = ci(boot)
    rec[name], rec[name + "_lo"], rec[name + "_hi"] = float(point), lo, hi


def fmt(rec, name):
    v = rec.get(name, np.nan)
    if v is None or pd.isna(v):
        return "-"
    return f"{v:.2f} ({rec[name + '_lo']:.2f}-{rec[name + '_hi']:.2f})"


def show(title, df, cols):
    print(f"\n== {title} ==")
    t = pd.DataFrame({c: ([fmt(r, c) for r in df.to_dict("records")] if c + "_lo" in df.columns else df[c])
                      for c in cols})
    print(t.to_string(index=False))


# ---------------------------------------------------------------- loading
def last_ok(path):
    last = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                last[r["key"]] = r
    return {k: r for k, r in last.items() if not r.get("error")}


def gt_row(df, row):
    return (df[[vc.vd(v) for v in vc.EXP1_ROWS[row][0]]].max(axis=1) == 1).to_numpy()


# ---------------------------------------------------------------- Exp 1
def exp1_sensitivity(df, prev, idx, min_pos):
    out = []
    for row, (vlabs, cx) in vc.EXP1_ROWS.items():
        gt = gt_row(df, row)
        p = df[cx + "_pred"].to_numpy(float)
        rec = {"finding": row, "chexpert_class": cx, "n_pos": int(gt.sum()), "n_neg": int((~gt).sum()),
               "low_n": bool(gt.sum() < min_pos),
               "report_pos": int((gt & (p == 1)).sum()), "report_hedged": int((gt & (p == -1)).sum()),
               "report_denied": int((gt & (p == 0)).sum()), "report_omitted": int((gt & np.isnan(p)).sum())}
        put(rec, "sens_strict", *rate(gt & (p == 1), gt, idx))
        put(rec, "sens_lenient", *rate(gt & np.isin(p, [1, -1]), gt, idx))
        put(rec, "spec_strict", *rate(~gt & (p != 1), ~gt, idx))
        out.append(rec)
    res = pd.DataFrame(out)
    if prev is not None:
        res = res.merge(prev[prev["kind"] == "row"].drop(columns=["kind", "chexpert_class"]), on="finding", how="left")
    return res


def exp1_size(df, boxes, idx):
    """Recall (strict) by union box area bin, per Exp 1 row and pooled over lesion rows."""
    keys = df["image_id"].to_numpy()
    b = boxes[boxes["kind"] == "row"].set_index(["group", "image_id"])
    n = len(df)
    pooled = {}  # (variant, bin) -> per-image (hits, positives)
    out, nobox = [], {}
    for row, (_, cx) in vc.EXP1_ROWS.items():
        gt = gt_row(df, row)
        hit = df[cx + "_pred"].to_numpy(float) == 1
        sub = b.loc[row] if row in b.index.get_level_values(0) else pd.DataFrame(columns=["area_frac", "kept_frac"])
        area = sub["area_frac"].reindex(keys).to_numpy(float)
        kept = sub["kept_frac"].reindex(keys).to_numpy(float)
        nobox[row] = int((gt & np.isnan(area)).sum())
        bins = np.array([vc.size_bin(a) if not np.isnan(a) else None for a in area], dtype=object)
        for _, _, bn in vc.SIZE_BINS:
            inb = gt & (bins == bn)
            rec = {"finding": row, "bin": bn, "n_pos": int(inb.sum()), "hits": int((inb & hit).sum()),
                   "n_pos_mostly_cropped": int((inb & (kept < .5)).sum())}
            put(rec, "recall", *rate(inb & hit, inb, idx))
            out.append(rec)
            if row in vc.NOT_LESIONS:
                continue
            for variant, mask in (("all lesions", inb), ("lesions, >=50% inside crop", inb & (kept >= .5)),
                                  ("lesions, <50% inside crop", inb & (kept < .5))):
                h, d = pooled.get((variant, bn), (np.zeros(n), np.zeros(n)))
                pooled[(variant, bn)] = (h + (mask & hit), d + mask)
    for (variant, bn), (h, d) in pooled.items():
        rec = {"finding": f"POOLED: {variant}", "bin": bn, "n_pos": int(d.sum()), "hits": int(h.sum())}
        put(rec, "recall", *rate(h, d, idx))
        out.append(rec)
    print("\npositives without boxes (not in the size analysis): "
          + ", ".join(f"{k} {v}" for k, v in nobox.items() if v))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Exp 2
def yesno_matrix(recs, keys, field="p_yes"):
    return pd.DataFrame({k: {q: v[field] for q, v in recs[k]["probs"].items()} for k in keys}).T


def exp2_auroc(ym, P, M, prior, idx, min_pos):
    out = []
    for f in vc.YESNO_FINDINGS:
        q = f"plain__{vc.slug(f)}"
        y = (ym[vc.vd(f)] == 1).to_numpy()
        s = P[q].to_numpy(float)
        rec = {"finding": f, "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "low_n": bool(y.sum() < min_pos),
               "median_mass_yes_no": float(np.median(M[q]))}
        if y.sum() >= min_pos:
            put(rec, "auroc", *auroc_boot(s, y, idx))
        put(rec, "sens_at_0.5", *rate(y & (s > .5), y, idx))
        put(rec, "spec_at_0.5", *rate(~y & (s <= .5), ~y, idx))
        put(rec, "mean_p_yes_pos", *rate(s * y, y, idx))
        put(rec, "mean_p_yes_neg", *rate(s * ~y, ~y, idx))
        if prior is not None:
            noise = prior.loc[[k for k in prior.index if k.startswith("noise")], q]
            black = prior.loc["black", q]
            rec.update(prior_black=float(black), prior_noise_mean=float(noise.mean()),
                       prior_noise_sd=float(noise.std(ddof=1)) if len(noise) > 1 else np.nan)
            put(rec, "share_above_black_pos", *rate(y & (s > black), y, idx))
            put(rec, "share_above_black_neg", *rate(~y & (s > black), ~y, idx))
        out.append(rec)
    return pd.DataFrame(out)


def exp2_vs_report(df, P, idx):
    """Among GT positives: does the yes/no answer say yes while the report omits / denies / hedges?"""
    out = []
    for row, (vlabs, cx) in vc.EXP1_ROWS.items():
        qs = [f"plain__{vc.slug(v)}" for v in vlabs]
        s = P[qs].max(axis=1).to_numpy(float)  # row = any of its VinDr labels
        gt = gt_row(df, row)
        p = df[cx + "_pred"].to_numpy(float)
        yes = gt & (s > .5)
        miss = yes & ((p == 0) | np.isnan(p))
        rec = {"finding": row, "n_pos": int(gt.sum()), "pos_p_yes_gt_0.5": int(yes.sum()),
               "of_which_report_pos": int((yes & (p == 1)).sum()), "of_which_report_hedged": int((yes & (p == -1)).sum()),
               "of_which_report_denied": int((yes & (p == 0)).sum()),
               "of_which_report_omitted": int((yes & np.isnan(p)).sum()),
               "pos_p_yes_le_0.5_but_report_pos": int((gt & (s <= .5) & (p == 1)).sum())}
        put(rec, "share_yes_but_report_misses", *rate(miss, yes, idx))
        out.append(rec)
    return pd.DataFrame(out)


def exp2_priming(ym, P, M, idx):
    out = []
    for f in vc.GUIDED:
        y = (ym[vc.vd(f)] == 1).to_numpy()
        boots = {}
        for pr in PROMPTS:
            s = P[f"{pr}__{vc.slug(f)}"].to_numpy(float)
            rec = {"finding": f, "prompt": pr, "n_pos": int(y.sum()), "n_neg": int((~y).sum())}
            m = {"auroc": auroc_boot(s, y, idx), "sens_at_0.5": rate(y & (s > .5), y, idx),
                 "spec_at_0.5": rate(~y & (s <= .5), ~y, idx), "mean_p_yes_pos": rate(s * y, y, idx),
                 "mean_p_yes_neg": rate(s * ~y, ~y, idx),
                 # mean P(yes or no) at the first token: a prompt that pushes the model out of yes/no answers
                 # (low mass) changes the answer format, not only what the model sees
                 "mean_mass_yes_no": rate(M[f"{pr}__{vc.slug(f)}"].to_numpy(float), np.ones(len(y)), idx)}
            for k, (pt, b) in m.items():
                put(rec, k, pt, b)
            boots[pr] = m
            out.append(rec)
        for a, b in (("guided", "plain"), ("control", "plain"), ("guided", "control")):
            rec = {"finding": f, "prompt": f"delta {a} - {b}", "n_pos": int(y.sum()), "n_neg": int((~y).sum())}
            for k in boots[a]:
                put(rec, k, boots[a][k][0] - boots[b][k][0], boots[a][k][1] - boots[b][k][1])
            out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures
INK, INK2, GRID = "#191919", "#666666", "#e5e5e5"
S1, S2, S3 = "#257de0", "#f47230", "#00b894"  # categorical slots 1-3 (dataviz reference palette)


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def _labels(ax, x, y, texts, dy=.045, near=.12):
    """Direct labels right of the points; labels of nearby points (|log10 dx| < near) are pushed apart
    vertically, with a thin leader line when a label moved."""
    placed = []
    for i in np.argsort(y):
        ly = y[i]
        while any(abs(np.log10(x[i] / px)) < near and abs(ly - py) < dy for px, py in placed):
            ly += dy
        placed.append((x[i], ly))
        ax.annotate(texts[i], (x[i], y[i]), xytext=(x[i] * 1.04, ly), textcoords="data", fontsize=8, color=INK,
                    va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=.6) if ly != y[i] else None)


def fig_exp1(sens, path):
    import matplotlib.pyplot as plt
    panels = [(c, t) for c, t in (("vindr_train_prev_majority", "VinDr train (2 of 3 radiologists)"),
                                  ("mimic_train_prev", "MIMIC-CXR train (CheXpert labels)")) if c in sens.columns]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5), squeeze=False)
    pooled_sens = sens["report_pos"].sum() / sens["n_pos"].sum()
    for ax, (col, title) in zip(axes[0], panels):
        d = sens[(sens["n_pos"] > 0) & (sens[col] > 0)]
        for lown, face in ((False, S1), (True, "white")):
            e = d[d["low_n"] == lown]
            ax.errorbar(e[col], e["sens_strict"], yerr=[e["sens_strict"] - e["sens_strict_lo"],
                        e["sens_strict_hi"] - e["sens_strict"]], fmt="o", ms=8, color=S1, mfc=face,
                        mec=S1, ecolor=S1, elinewidth=1.2, capsize=0,
                        label="n_pos < 10 (hollow)" if lown else "n_pos >= 10")
        _labels(ax, d[col].to_numpy(float), d["sens_strict"].to_numpy(float),
                [f"{f} ({n})" for f, n in zip(d["finding"], d["n_pos"])])
        ax.axvline(d[col].median(), color=INK2, ls="--", lw=1)
        ax.axhline(pooled_sens, color=INK2, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(f"training prevalence, {title} (log)", color=INK)
        ax.set_ylabel("sensitivity from generated reports (strict)", color=INK)
        _style(ax)
        ax.text(.02, .02, "low prev + low sens: rare\nhigh prev + low sens: subtle", transform=ax.transAxes,
                fontsize=8, color=INK2, va="bottom")
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("VinDr test: sensitivity vs training prevalence (dashed: median prevalence, pooled sensitivity)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_size(size, path):
    import matplotlib.pyplot as plt
    variants = [("POOLED: all lesions", S1), ("POOLED: lesions, >=50% inside crop", S2)]
    bins = [b for _, _, b in vc.SIZE_BINS]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    w = .36
    for j, (v, c) in enumerate(variants):
        d = size[size["finding"] == v].set_index("bin").reindex(bins)
        x = np.arange(len(bins)) + (j - .5) * w
        ax.bar(x, d["recall"], width=w - .03, color=c, label=v.replace("POOLED: ", ""))
        ax.errorbar(x, d["recall"], yerr=[d["recall"] - d["recall_lo"], d["recall_hi"] - d["recall"]],
                    fmt="none", ecolor=INK2, elinewidth=1.2)
        for xi, (_, r) in zip(x, d.iterrows()):
            ax.text(xi, 0.01, f"n={int(r['n_pos']) if not pd.isna(r['n_pos']) else 0}", ha="center", va="bottom",
                    fontsize=8, color="white")
    ax.set_xticks(np.arange(len(bins)), [f"box area {b} of image" for b in bins])
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall from generated reports (strict)", color=INK)
    ax.set_title("Recall by lesion size (cardiomegaly / aortic enlargement excluded)", color=INK, fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_auroc(auc, path):
    import matplotlib.pyplot as plt
    d = auc.dropna(subset=["auroc"]).sort_values("auroc")
    fig, ax = plt.subplots(figsize=(7.5, max(3, .3 * len(d) + 1.2)))
    y = np.arange(len(d))
    ax.errorbar(d["auroc"], y, xerr=[d["auroc"] - d["auroc_lo"], d["auroc_hi"] - d["auroc"]], fmt="o", ms=7,
                color=S1, elinewidth=1.2)
    ax.set_yticks(y, [f"{f} ({n})" for f, n in zip(d["finding"], d["n_pos"])])
    ax.axvline(.5, color=INK2, ls="--", lw=1)
    ax.set_xlabel("AUROC of p_yes (plain question), 95% CI", color=INK)
    ax.set_title("VinDr test: yes/no AUROC per finding (n_pos)", color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_priming(prim, path):
    """Per finding: AUROC, sensitivity / specificity at 0.5 and yes/no mass for plain / guided / control prompts."""
    import matplotlib.pyplot as plt
    fs, metrics = list(vc.GUIDED), [("auroc", "AUROC"), ("sens_at_0.5", "sensitivity @0.5"),
                                    ("spec_at_0.5", "specificity @0.5"), ("mean_mass_yes_no", "P(yes or no), mean")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 1.1 * len(fs) + 1.6), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        for j, (pr, c) in enumerate(zip(PROMPTS, (S1, S2, S3))):
            e = prim[prim["prompt"] == pr].set_index("finding").reindex(fs)
            yy = np.arange(len(fs)) + (1 - j) * .22
            ax.errorbar(e[m], yy, xerr=[e[m] - e[m + "_lo"], e[m + "_hi"] - e[m]], fmt="o", ms=7, color=c,
                        elinewidth=1.2, label=pr)
        ax.set_yticks(np.arange(len(fs)), fs)
        ax.set_ylim(-.6, len(fs) - .4)
        ax.set_xlim(0, 1.02)
        ax.set_title(title, color=INK, fontsize=10)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Priming: plain vs guided vs length-matched control prompt (95% CI)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(REPO, "manifests", "vindr.csv"))
    ap.add_argument("--boxes", default=os.path.join(REPO, "manifests", "vindr_boxes.csv"))
    ap.add_argument("--pred", default=os.path.join(RESULTS_LABELS, "vindr_pred_labels.csv"))
    ap.add_argument("--prev", default=os.path.join(RESULTS_SCORES, "vindr_training_prevalence.csv"))
    ap.add_argument("--yesno", default=os.path.join(RESULTS_INF, "vindr_yesno.jsonl"))
    ap.add_argument("--prior", default=os.path.join(RESULTS_INF, "vindr_yesno_prior.jsonl"))
    ap.add_argument("--out", default=RESULTS_SCORES)
    ap.add_argument("--figs", default=RESULTS_FIGS)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_pos", type=int, default=10, help="AUROC / 'low_n' flag threshold")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)
    have = lambda p: os.path.exists(p) or print(f"(missing, skipped: {p})")  # noqa: E731

    man = pd.read_csv(args.manifest)
    prev = pd.read_csv(args.prev) if have(args.prev) else None
    df = None
    if have(args.pred):
        pred = pd.read_csv(args.pred)
        df = man.merge(pred[["key"] + LABELS], on="key", how="inner", suffixes=("", "_pred"))
        assert len(df) == len(man) == len(pred), f"key mismatch: manifest {len(man)}, pred {len(pred)}, matched {len(df)}"
        idx = np.random.default_rng(args.seed).integers(0, len(df), (args.n_boot, len(df)))
        print(f"\nVinDr test images: {len(df)}. NOTE: ~1,103 VinDr test images were probably in CXR-LLaVA's "
              f"training/validation data (paper uses 16,103 VinDr images; the public train split has 15,000).")

        sens = exp1_sensitivity(df, prev, idx, args.min_pos)
        sens.to_csv(os.path.join(args.out, "vindr_exp1_sensitivity.csv"), index=False)
        cols = ["finding", "n_pos", "sens_strict", "sens_lenient", "spec_strict", "report_pos", "report_hedged",
                "report_denied", "report_omitted"] + [c for c in ("vindr_train_prev_majority", "mimic_train_prev")
                                                      if c in sens.columns]
        show("Exp 1: sensitivity from reports (strict = labeler 1; lenient = 1 or -1)", sens, cols)
        fig_exp1(sens, os.path.join(args.figs, "vindr_exp1_prevalence_vs_sensitivity.png"))

        if have(args.boxes):
            size = exp1_size(df, pd.read_csv(args.boxes), idx)
            size.to_csv(os.path.join(args.out, "vindr_exp1_size.csv"), index=False)
            show("Exp 1: recall by union box area (strict)", size[size["n_pos"] > 0],
                 ["finding", "bin", "n_pos", "hits", "recall"])
            fig_size(size, os.path.join(args.figs, "vindr_exp1_size.png"))

    if have(args.yesno):
        recs = last_ok(args.yesno)
        ym = man[man["key"].isin(recs)].reset_index(drop=True)
        print(f"\nyes/no results for {len(ym)}/{len(man)} manifest images")
        P, M = yesno_matrix(recs, ym["key"]), yesno_matrix(recs, ym["key"], "mass_yes_no")
        idx2 = np.random.default_rng(args.seed).integers(0, len(ym), (args.n_boot, len(ym)))
        prior = None
        if have(args.prior):
            pr = last_ok(args.prior)
            prior = yesno_matrix(pr, list(pr))
        auc = exp2_auroc(ym, P, M, prior, idx2, args.min_pos)
        auc.to_csv(os.path.join(args.out, "vindr_exp2_auroc.csv"), index=False)
        show("Exp 2: yes/no per finding (plain question)", auc,
             ["finding", "n_pos", "auroc", "sens_at_0.5", "spec_at_0.5", "mean_p_yes_pos", "mean_p_yes_neg"]
             + (["prior_black", "prior_noise_mean", "share_above_black_neg"] if prior is not None else [])
             + ["median_mass_yes_no"])
        prim = exp2_priming(ym, P, M, idx2)
        prim.to_csv(os.path.join(args.out, "vindr_exp2_priming.csv"), index=False)
        show("Exp 2: priming (deltas are paired bootstrap)", prim,
             ["finding", "prompt", "n_pos", "auroc", "sens_at_0.5", "spec_at_0.5", "mean_p_yes_pos", "mean_p_yes_neg",
              "mean_mass_yes_no"])
        fig_auroc(auc, os.path.join(args.figs, "vindr_exp2_auroc.png"))
        fig_priming(prim, os.path.join(args.figs, "vindr_exp2_priming.png"))

        if df is not None:
            both = df[df["key"].isin(recs)].reset_index(drop=True)
            Pb = P.loc[both["key"]].reset_index(drop=True)
            idx3 = np.random.default_rng(args.seed).integers(0, len(both), (args.n_boot, len(both)))
            vs = exp2_vs_report(both, Pb, idx3)
            vs.to_csv(os.path.join(args.out, "vindr_exp2_vs_report.csv"), index=False)
            show("Exp 2: GT positives, yes/no says yes (p>0.5) but the report ...", vs,
                 ["finding", "n_pos", "pos_p_yes_gt_0.5", "of_which_report_pos", "of_which_report_hedged",
                  "of_which_report_denied", "of_which_report_omitted", "share_yes_but_report_misses",
                  "pos_p_yes_le_0.5_but_report_pos"])
    print(f"\n-> tables in {args.out} (vindr_*.csv), figures in {args.figs}")


if __name__ == "__main__":
    main()
