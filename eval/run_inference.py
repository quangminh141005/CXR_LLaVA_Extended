"""Step 5: generate reports with the 4-bit model for every image of the test manifests.

Model is loaded ONCE (4-bit NF4, same as the Step 2 smoke test), then each dataset runs
in the same process, one after the other, each into its own file:

  manifests/chexpert.csv -> results/inference/chexpert.jsonl
  manifests/mimic.csv    -> results/inference/mimic.jsonl

One JSON line per image: {key, dataset, report, n_tokens, seconds, error}, flushed after
every image. Decoding: greedy (temperature 0), prompt = write_radiologic_report default,
max 512 new tokens (generate_report()).

Resumable: images already in the JSONL without error are skipped; images that failed are
retried on the next run (the last line per key counts). Each run is recorded in
results/inference/run_info.json (git commit, settings, GPU, times).

Label task (CheXpert, paper Fig. 2b): --task labels --label_prompt gemini|gpt4v
  asks the model for 0/1 JSON labels instead of a report ->
  results/inference/<ds>_labels_<prompt>.jsonl  {key, dataset, output, n_tokens, seconds, error}
  (parse with: python eval/label_reports.py collect-json --label_prompt <prompt>)

Usage (on the server, repo root, inference env, in tmux; no CPU jobs in parallel):
    python -u eval/run_inference.py --limit 5 2>&1 | tee log/inference/inference_test.log   # quick test
    python -u eval/run_inference.py 2>&1 | tee -a log/inference/inference.log                # full run
    python -u eval/run_inference.py --datasets mimic 2>&1 | tee -a log/inference/inference.log
    python -u eval/run_inference.py --datasets vindr 2>&1 | tee -a log/inference/inference_vindr.log   # extension
    python -u eval/run_inference.py --task labels --label_prompt gemini --limit 20 2>&1 | tee log/inference/labels_smoke_gemini.log
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from label_json import parse_labels  # noqa: E402
from paths import RESULTS_INF  # noqa: E402

DATASETS = ["chexpert", "mimic"]  # default
EXTRA_DATASETS = ["vindr"]  # extension experiments, only when named
ETA_EVERY = 25


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def git_info():
    def run(*cmd):
        try:
            return subprocess.check_output(["git", "-C", REPO, *cmd], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = run("status", "--porcelain", "--", "eval", "CXR_LLAVA_HF")
    return {"commit": run("rev-parse", "HEAD"), "code_modified": bool(status) if status is not None else None}


def read_done(path):
    """Keys whose LAST record has no error (failed images are retried)."""
    last = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    last[r["key"]] = r
    return {k for k, r in last.items() if not r.get("error")}, sum(1 for r in last.values() if r.get("error"))


def out_file(args, ds):
    name = f"{ds}.jsonl" if args.task == "report" else f"{ds}_labels_{args.label_prompt}.jsonl"
    return os.path.join(args.out, name)


def plan(args):
    """Per dataset: manifest rows still to do (respecting --limit)."""
    todo = {}
    for ds in args.datasets:
        man = pd.read_csv(os.path.join(args.manifests, f"{ds}.csv"))
        if args.limit:
            man = man.head(args.limit)
        done, n_err = read_done(out_file(args, ds))
        todo[ds] = man[~man["key"].isin(done)]
        print(f"{ds:9s} manifest rows {len(man):5d}  done {len(man) - len(todo[ds]):5d}  "
              f"to do {len(todo[ds]):5d}" + (f"  (incl. {n_err} earlier errors to retry)" if n_err else ""))
    return todo


def save_run_info(path, info):
    runs = []
    if os.path.exists(path):
        with open(path) as f:
            runs = json.load(f)
    runs = [r for r in runs if r.get("run_id") != info["run_id"]] + [info]
    with open(path, "w") as f:
        json.dump(runs, f, indent=2)


def run_dataset(model, ds, rows, out_path, args):
    import torch
    from PIL import Image

    times, n_err, t_ds = [], 0, time.perf_counter()
    field = "report" if args.task == "report" else "output"
    with open(out_path, "a") as f:
        for i, (_, r) in enumerate(rows.iterrows(), 1):
            rec = {"key": r["key"], "dataset": r["dataset"]}
            t = time.perf_counter()
            try:
                with Image.open(r["image_path"]) as im:
                    im.load()
                    if args.task == "report":
                        text = model.generate_report(im, temperature=0.0)
                    else:
                        text = model.generate_labels(im, prompt=args.label_prompt, temperature=0.0)
                rec.update({field: text}, n_tokens=len(model.tokenizer(text).input_ids), error=None)
                if not text:
                    rec["error"] = f"empty {field}"
            except Exception as e:  # keep going; retried on the next run
                rec.update({field: None}, n_tokens=0, error=f"{type(e).__name__}: {e}")
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            rec["seconds"] = round(time.perf_counter() - t, 2)
            f.write(json.dumps(rec) + "\n")
            f.flush()

            if rec["error"]:
                n_err += 1
            else:
                times.append(rec["seconds"])
            status = f"ERROR {rec['error']}" if rec["error"] else f"{rec['n_tokens']} tok"
            if args.task == "labels" and not rec["error"]:
                lab, method = parse_labels(rec["output"])
                status += f"  parsed {sum(v is not None for v in lab.values())}/14 ({method})"
            print(f"[{ds} {i}/{len(rows)}] {r['key']}  {rec['seconds']:.1f} s  {status}")
            if i % ETA_EVERY == 0 and times:
                mean = sum(times) / len(times)
                eta = datetime.timedelta(seconds=int(mean * (len(rows) - i)))
                print(f"    -- mean {mean:.1f} s/image, ETA {ds}: {eta}, errors so far {n_err}")

    mean = sum(times) / len(times) if times else float("nan")
    return {"dataset": ds, "images": len(rows), "ok": len(times), "errors": n_err,
            "mean_s_per_image": round(mean, 2), "wall_minutes": round((time.perf_counter() - t_ds) / 60, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["report", "labels"], default="report",
                    help="report: free-text report (MIMIC protocol); labels: 0/1 JSON (CheXpert protocol)")
    ap.add_argument("--label_prompt", choices=["gemini", "gpt4v"], default="gemini",
                    help="label task only: which supplement prompt (see CXRLLAVAModel.LABEL_PROMPTS)")
    ap.add_argument("--datasets", nargs="+", choices=DATASETS + EXTRA_DATASETS, default=None,
                    help="default: both for report, chexpert for labels")
    ap.add_argument("--limit", type=int, default=None, help="only the first N manifest rows per dataset")
    ap.add_argument("--manifests", default=os.path.join(REPO, "manifests"))
    ap.add_argument("--out", default=RESULTS_INF)
    ap.add_argument("--weights", default=os.path.join(REPO, "weights", "CXR-LLAVA-v2"))
    args = ap.parse_args()
    if args.datasets is None:
        args.datasets = DATASETS if args.task == "report" else ["chexpert"]
    os.makedirs(args.out, exist_ok=True)

    todo = plan(args)
    if all(t.empty for t in todo.values()):
        print("nothing to do")
        return

    import torch
    from load_model import check_model, load_model

    assert torch.cuda.is_available(), "CUDA not available"
    info = {"run_id": datetime.datetime.now().isoformat(timespec="microseconds"), "start": now(), "end": None, "datasets": args.datasets, "limit": args.limit,
            **git_info(), "weights": os.path.abspath(args.weights),
            "gpu": torch.cuda.get_device_name(0),
            "model": "LLaMA 4-bit NF4 (compute fp16, no double quant); vision tower, mm_projector, lm_head fp16",
            "task": args.task, "label_prompt": args.label_prompt if args.task == "labels" else None,
            "decoding": "greedy (temperature 0), max_new_tokens 512, prompt "
                        + ("write_radiologic_report default" if args.task == "report"
                           else f"label prompt '{args.label_prompt}' (generate_labels)"),
            "per_dataset": []}
    info_path = os.path.join(args.out, "run_info.json")
    save_run_info(info_path, info)
    print(f"\nrun start {info['start']}  commit {info['commit']}  code modified: {info['code_modified']}")

    t0 = time.perf_counter()
    model = load_model(args.weights)
    print(f"model loaded in {time.perf_counter() - t0:.0f} s on {info['gpu']}")
    check_model(model)
    torch.cuda.reset_peak_memory_stats()

    for ds in args.datasets:
        if todo[ds].empty:
            continue
        print(f"\n== {ds}: {len(todo[ds])} images -> {out_file(args, ds)}")
        summ = run_dataset(model, ds, todo[ds], out_file(args, ds), args)
        info["per_dataset"].append(summ)
        save_run_info(info_path, info)
        print(f"== {ds} done: {summ}")

    info["end"] = now()
    info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    save_run_info(info_path, info)
    print(f"\nrun end {info['end']}, peak VRAM {info['peak_vram_gb']} GB")
    print("status now:")
    plan(args)


if __name__ == "__main__":
    main()
