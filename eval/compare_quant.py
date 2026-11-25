"""Step 2.5 B: compare 4-bit (GPU) reports with the full-precision fp32 (CPU) reference.

Three sub-commands, all writing to --out (default checks/quant_check/):

  select   pick the comparison set once -> selection.csv
           12 images per dataset = 6 enriched (one image positive for each of the 6
           findings) + 6 random from the test pool. Seed 0. Labels are only used to
           PICK images; the comparison itself is model vs model.
  run      generate reports for every image in selection.csv -> <mode>.jsonl
           --mode 4bit     GPU, NF4 (same as Step 2)
           --mode fp32cpu  CPU, unquantized fp32 (reference, see Step 2.5 A)
           Resumable: images already in the JSONL are skipped.
  compare  pair the two JSONL files -> paired.csv + side_by_side.txt + summary
           (exact match, word-level similarity). Label agreement comes later
           from the CheXbert labeler (Step 6.1) run on paired.csv.

Usage (on the server, repo root):
    python -u eval/compare_quant.py select
    python -u eval/compare_quant.py run --mode 4bit 2>&1 | tee checks/quant_check/run_4bit.log
    numactl --cpunodebind=0 --membind=0 \
      python -u eval/compare_quant.py run --mode fp32cpu 2>&1 | tee checks/quant_check/run_fp32cpu.log
    python -u eval/compare_quant.py compare
"""
import argparse
import difflib
import json
import os
import re
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from check_data import DEFAULTS, FINDINGS  # noqa: E402  (same paths/findings as Step 0)
from paths import CHECKS  # noqa: E402

MIMIC_JPG_ROOT = "/storage/shared/MIMIC-CXR-JPG/split-4/files"
MIMIC_LABELS = ("/storage/shared/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0/"
                "mimic-cxr-2.0.0-chexpert.csv.gz")
DEFAULT_OUT = os.path.join(CHECKS, "quant_check")
N_ENRICHED, N_RANDOM = 6, 6
FALLBACK_FINDING = "Pneumothorax"  # used when a finding has no positive in a dataset (VinDr Edema)


# ------------------------------------------------------------------ pools (test-set rules of Step 0)
def pool_chexpert():
    gt = pd.read_csv(os.path.join(DEFAULTS["chexpert_root"], "groundtruth.csv"))
    gt["image_id"] = gt["Study"].str.extract(r"(patient\d+/study\d+)")[0]
    gt = gt.dropna(subset=["image_id"])
    gt["image_path"] = [os.path.join(DEFAULTS["chexpert_root"], "CheXpert", "test", "test", k, "view1_frontal.jpg")
                        for k in gt["image_id"]]
    return gt[["image_id", "image_path"] + FINDINGS], "groundtruth.csv"


def pool_vindr():
    lab = pd.read_csv(DEFAULTS["vindr_labels"]).drop_duplicates("image_id")
    cols = {f: next((c for c in lab.columns if c.lower() == f.lower()), None) for f in FINDINGS}
    out = pd.DataFrame({"image_id": lab["image_id"]})
    out["image_path"] = [os.path.join(DEFAULTS["vindr_png_dir"], f"{i}.png") for i in out["image_id"]]
    for f, c in cols.items():
        out[f] = lab[c].values if c else float("nan")
    return out, "image_labels_test.csv"


def pool_mimic():
    rows = pd.read_csv(DEFAULTS["mimic_csv"], header=None)[0]
    rels = [r.strip().replace("\\", "/").removeprefix("files/").replace(".dcm", ".jpg") for r in rows]
    out = pd.DataFrame({"rel": rels})
    out["image_id"] = [os.path.splitext(os.path.basename(r))[0] for r in rels]
    out["study_id"] = [int(re.search(r"/s(\d+)/", r).group(1)) for r in rels]
    out["image_path"] = [os.path.join(MIMIC_JPG_ROOT, r) for r in rels]
    lab = pd.read_csv(MIMIC_LABELS)[["study_id"] + FINDINGS]
    out = out.merge(lab, on="study_id", how="left")
    return out[["image_id", "image_path"] + FINDINGS], "mimic-cxr-2.0.0-chexpert (1 = positive; -1/blank not used)"


def pool_nlmcxr():
    root = DEFAULTS["indiana_root"]
    rep = pd.read_csv(os.path.join(root, "indiana_reports.csv"))
    proj = pd.read_csv(os.path.join(root, "indiana_projections.csv"))
    has_text = lambda s: s.fillna("").astype(str).str.strip().str.len() > 0  # noqa: E731
    rep = rep[has_text(rep["findings"]) | has_text(rep["impression"])]
    fr = proj[proj["projection"].str.lower() == "frontal"].sort_values("filename")
    first = fr.groupby("uid")["filename"].first()
    rep = rep[rep["uid"].isin(first.index)].copy()
    rep["image_id"] = [first[u] for u in rep["uid"]]
    rep["image_path"] = [os.path.join(root, "images", "images_normalized", f) for f in rep["image_id"]]
    if "Problems" in rep.columns:
        probs = rep["Problems"].fillna("").str.lower()
        for f in FINDINGS:
            rep[f] = probs.str.contains(f.lower(), regex=False).astype(int)
        src = "indiana_reports.csv Problems column (weak labels)"
    else:
        for f in FINDINGS:
            rep[f] = float("nan")
        src = "NONE (no Problems column) -> enriched part is random"
    return rep[["image_id", "image_path"] + FINDINGS], src


POOLS = {"CheXpert": pool_chexpert, "VinDr": pool_vindr, "NLMCXR": pool_nlmcxr, "MIMIC": pool_mimic}


# ------------------------------------------------------------------ select
def pick(pool, seed):
    """6 enriched (one positive per finding) + 6 random, no duplicates."""
    chosen, rows = set(), []
    for k, f in enumerate(FINDINGS):
        target = f
        cand = pool[(pool[f] == 1) & ~pool["image_id"].isin(chosen)]
        if cand.empty:  # e.g. VinDr has no Edema positive
            target = f"{f}->{FALLBACK_FINDING}"
            cand = pool[(pool[FALLBACK_FINDING] == 1) & ~pool["image_id"].isin(chosen)]
        if cand.empty:  # no labels at all (NLMCXR without Problems) -> random
            target = f"{f}->random"
            cand = pool[~pool["image_id"].isin(chosen)]
        r = cand.sample(1, random_state=seed + k).iloc[0]
        chosen.add(r["image_id"])
        rows.append({**r.to_dict(), "group": "enriched", "target": target})
    rest = pool[~pool["image_id"].isin(chosen)].sample(N_RANDOM, random_state=seed + 100)
    rows += [{**r.to_dict(), "group": "random", "target": ""} for _, r in rest.iterrows()]
    return rows


def cmd_select(args):
    sel_path = os.path.join(args.out, "selection.csv")
    if os.path.exists(sel_path) and not args.force:
        sys.exit(f"{sel_path} exists (use --force to re-select; existing JSONL results would no longer match)")
    rows = []
    for ds, fn in POOLS.items():
        pool, src = fn()
        picked = pick(pool, args.seed)
        for r in picked:
            r["dataset"], r["gt_source"] = ds, src
        rows += picked
        n_pos = {f: int((pool[f] == 1).sum()) for f in FINDINGS}
        print(f"{ds:9s} pool {len(pool):5d}  positives {n_pos}\n          labels: {src}")
    sel = pd.DataFrame(rows)
    sel["key"] = sel["dataset"] + ":" + sel["image_id"].astype(str)
    sel = sel[["key", "dataset", "group", "target", "image_id", "image_path", "gt_source"] + FINDINGS]
    missing = [p for p in sel["image_path"] if not os.path.exists(p)]
    sel.to_csv(sel_path, index=False)
    print(f"\nselected {len(sel)} images -> {sel_path}")
    print(sel.groupby(["dataset", "group"]).size().to_string())
    if missing:
        print(f"\n!! {len(missing)} selected files do not exist:\n  " + "\n  ".join(missing))


# ------------------------------------------------------------------ run
def cmd_run(args):
    import torch
    from PIL import Image
    from load_model import load_model, load_model_cpu

    sel = pd.read_csv(os.path.join(args.out, "selection.csv"))
    out_path = os.path.join(args.out, f"{args.mode}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = sel[~sel["key"].isin(done)]
    print(f"mode {args.mode}: {len(sel)} selected, {len(done)} done, {len(todo)} to do -> {out_path}")
    if todo.empty:
        return

    t0 = time.perf_counter()
    if args.mode == "4bit":
        model = load_model(args.weights)
    else:
        torch.set_num_threads(args.threads)
        model = load_model_cpu(args.weights, "fp32")
    print(f"model loaded in {time.perf_counter() - t0:.0f} s")

    with open(out_path, "a") as f:
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            rec = {"key": r["key"], "mode": args.mode}
            t = time.perf_counter()
            try:
                report = model.generate_report(Image.open(r["image_path"]), temperature=0.0)
                rec.update(report=report, n_tokens=len(model.tokenizer(report).input_ids), error=None)
            except Exception as e:  # keep going; failed images can be retried by deleting their line
                rec.update(report=None, n_tokens=0, error=f"{type(e).__name__}: {e}")
            rec["seconds"] = round(time.perf_counter() - t, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            status = "ERROR " + rec["error"] if rec["error"] else f"{rec['n_tokens']} tok"
            print(f"[{i}/{len(todo)}] {r['key']}  {rec['seconds']} s  {status}")


# ------------------------------------------------------------------ compare
def norm(t):
    return " ".join(str(t).lower().split())


def word_sim(a, b):
    return difflib.SequenceMatcher(None, norm(a).split(), norm(b).split()).ratio()


def cmd_compare(args):
    sel = pd.read_csv(os.path.join(args.out, "selection.csv"))
    res = {}
    for mode in ("4bit", "fp32cpu"):
        with open(os.path.join(args.out, f"{mode}.jsonl")) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        res[mode] = pd.DataFrame(recs).drop_duplicates("key", keep="last").set_index("key")

    df = sel.set_index("key")
    for mode in res:
        df[f"report_{mode}"] = res[mode]["report"]
        df[f"seconds_{mode}"] = res[mode]["seconds"]
    ok = df["report_4bit"].notna() & df["report_fp32cpu"].notna()
    print(f"pairs with both reports: {int(ok.sum())}/{len(df)}")
    df = df[ok].copy()
    df["exact_match"] = [norm(a) == norm(b) for a, b in zip(df["report_4bit"], df["report_fp32cpu"])]
    df["word_similarity"] = [round(word_sim(a, b), 3) for a, b in zip(df["report_4bit"], df["report_fp32cpu"])]
    df.reset_index().to_csv(os.path.join(args.out, "paired.csv"), index=False)

    with open(os.path.join(args.out, "side_by_side.txt"), "w") as f:
        for key, r in df.sort_values("word_similarity").iterrows():
            gt = ", ".join(x for x in FINDINGS if r[x] == 1) or "none"
            f.write(f"=== {key}  [{r['group']} {r['target']}]  GT+: {gt}  "
                    f"exact={r['exact_match']}  sim={r['word_similarity']}\n"
                    f"--- fp32: {r['report_fp32cpu']}\n--- 4bit: {r['report_4bit']}\n\n")

    print("\n== Text agreement (label agreement comes from the CheXbert step) ==")
    summ = df.groupby(["group", "dataset"]).agg(n=("exact_match", "size"), exact_match=("exact_match", "mean"),
                                                 word_sim_mean=("word_similarity", "mean"),
                                                 word_sim_min=("word_similarity", "min"))
    print(summ.round(2).to_string())
    print(f"\nall: n={len(df)}  exact match {df['exact_match'].mean():.0%}  "
          f"word similarity mean {df['word_similarity'].mean():.2f}  min {df['word_similarity'].min():.2f}")
    print(f"s/image: 4bit {df['seconds_4bit'].mean():.1f}, fp32cpu {df['seconds_fp32cpu'].mean():.1f}")
    print(f"\nwritten: paired.csv, side_by_side.txt (least similar first) in {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["4bit", "fp32cpu"], required=True)
    r.add_argument("--weights", default=os.path.join(REPO, "weights", "CXR-LLAVA-v2"))
    r.add_argument("--threads", type=int, default=20)
    sub.add_parser("compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"select": cmd_select, "run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
