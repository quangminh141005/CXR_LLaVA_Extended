"""Extension Exp 2: yes/no probabilities on the VinDr test set (4-bit model, same as run_inference.py).

For every image, every question of eval/vindr_common.py (plain question per finding + guided and
length-matched control prompts for Pneumothorax and Nodule/Mass) -> p_yes = P(yes) / (P(yes) + P(no))
at the first answer token (CXRLLAVAModel.yes_no_probs; [system + image] KV cache reused per question).

Sub-commands:
  smoke   first N manifest images. Checks: cached vs full-prompt logits, yes+no probability mass,
          argmax token, and the greedy short answer of a few plain questions. -> checks/yesno_smoke/
  real    all manifest images -> results/inference/vindr_yesno.jsonl (resumable, last line per key)
  prior   image-free baseline: black image + N uniform-noise images (512x512, seeds 0..N-1)
          -> results/inference/vindr_yesno_prior.jsonl

Usage (on the server, repo root, inference env, in tmux):
    python -u eval/run_yesno.py smoke --n 20 2>&1 | tee log/inference/yesno_smoke.log
    python -u eval/run_yesno.py smoke --n 20 --priming_only 2>&1 | tee log/inference/yesno_smoke_priming.log
    python -u eval/run_yesno.py prior 2>&1 | tee log/inference/yesno_prior.log
    python -u eval/run_yesno.py real 2>&1 | tee -a log/inference/yesno_vindr.log
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from paths import CHECKS, RESULTS_INF  # noqa: E402
from run_inference import git_info, read_done, save_run_info  # noqa: E402
import vindr_common as vc  # noqa: E402

ETA_EVERY = 50
SMOKE_GREEDY = ["plain__pleural_effusion", "plain__cardiomegaly", "plain__pneumothorax", "plain__nodule_mass"]
PRIMING_GREEDY = ["guided__pneumothorax", "control__pneumothorax", "guided__nodule_mass", "control__nodule_mass"]
MIN_MASS = 0.8  # acceptance: median P(yes or no) per question on the smoke images


def priming_questions(qs):
    """the 6 guided/control questions + the 2 plain ones they are compared with"""
    keep = {f"{p}__{vc.slug(f)}" for f in vc.GUIDED for p in ("plain", "guided", "control")}
    return {k: v for k, v in qs.items() if k in keep}


def mass_table(rows):
    """per question: median / min P(yes or no), median p_yes, most common first tokens"""
    from collections import Counter, defaultdict
    m, py, top = defaultdict(list), defaultdict(list), defaultdict(Counter)
    for r in rows:
        for q, v in r.items():
            m[q].append(v["mass_yes_no"])
            py[q].append(v["p_yes"])
            top[q][v["top_token"]] += 1
    print(f"\n{'question':28s} mass_med mass_min p_yes_med  top first tokens")
    failed = []
    for q in sorted(m, key=lambda q: np.median(m[q])):
        med = float(np.median(m[q]))
        flag = "" if med >= MIN_MASS else f"  <-- below {MIN_MASS}"
        failed += [q] if flag else []
        print(f"{q:28s} {med:8.3f} {min(m[q]):8.3f} {np.median(py[q]):9.2f}  {dict(top[q].most_common(3))}{flag}")
    print(f"questions with median mass < {MIN_MASS}: {failed if failed else 'none'}")


def load(args):
    import torch
    from load_model import check_model, load_model
    assert torch.cuda.is_available(), "CUDA not available"
    t0 = time.perf_counter()
    model = load_model(args.weights)
    print(f"model loaded in {time.perf_counter() - t0:.0f} s on {torch.cuda.get_device_name(0)}")
    check_model(model)
    return model


def show_tokens(model, qs):
    yes, no = model._yes_no_ids()
    print("yes token ids: " + ", ".join(f"{i}={model.tokenizer.convert_ids_to_tokens(i)!r}" for i in yes))
    print("no  token ids: " + ", ".join(f"{i}={model.tokenizer.convert_ids_to_tokens(i)!r}" for i in no))
    assert yes and no, "no yes/no tokens found in the vocabulary"
    n = {k: len(model.tokenizer(v).input_ids) for k, v in qs.items()}
    for f in vc.GUIDED:
        s = vc.slug(f)
        print(f"question tokens {f}: plain {n['plain__' + s]}, guided {n['guided__' + s]}, control {n['control__' + s]}")
    chat = model.apply_chat_template(model._yes_no_chat(qs["plain__pneumothorax"]))
    print(f"full prompt (template applied):\n{chat!r}")


def write_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def cmd_smoke(args, model, qs):
    from PIL import Image
    out_dir = os.path.join(CHECKS, "yesno_smoke")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke_priming.jsonl" if args.priming_only else "smoke.jsonl")
    open(out_path, "w").close()
    man = pd.read_csv(args.manifest).head(args.n)
    greedy_qs = PRIMING_GREEDY if args.priming_only else SMOKE_GREEDY
    diffs, masses, agree, n_cmp, all_probs = [], [], 0, 0, []
    for i, (_, r) in enumerate(man.iterrows(), 1):
        with Image.open(r["image_path"]) as im:
            im.load()
            t = time.perf_counter()
            res = model.yes_no_probs(im, qs, check_full=True)
            dt = time.perf_counter() - t
            greedy = {q: model._generate_text(model._yes_no_chat(qs[q]), im, temperature=0.0, max_new_tokens=8)
                      for q in greedy_qs}
        all_probs.append(res)
        diffs += [v["max_abs_logit_diff"] for v in res.values()]
        masses += [v["mass_yes_no"] for v in res.values()]
        for q, text in greedy.items():
            n_cmp += 1
            first = text.strip().lower()[:3]
            pred = "yes" if res[q]["p_yes"] > 0.5 else "no"
            agree += first.startswith(pred[:2])
        write_jsonl(out_path, {"key": r["key"], "seconds_with_full_check": round(dt, 2), "probs": res, "greedy": greedy})
        print(f"[{i}/{len(man)}] {r['key']} {dt:.1f} s (incl. full-prompt check) | " + " | ".join(
            f"{q} p={res[q]['p_yes']:.2f} greedy={greedy[q]!r}" for q in greedy_qs))
    diffs, masses = np.array(diffs), np.array(masses)
    print("\n== smoke summary ==")
    print(f"cached vs full logits: max abs diff max {diffs.max():.4f}, median {np.median(diffs):.4f} "
          f"(fp16 noise is ~1e-2; >0.5 would mean the cache path is wrong)")
    print(f"P(yes or no) at the first token: min {masses.min():.3f}, median {np.median(masses):.3f}, "
          f"share < 0.9: {(masses < 0.9).mean():.2%}")
    print(f"p_yes > 0.5 agrees with the greedy answer: {agree}/{n_cmp} "
          "(only answers starting with yes/no can agree; 'There is ...' counts as disagreement)")
    mass_table(all_probs)
    print(f"-> {out_path}")


def run_images(model, qs, items, out_path, label):
    """items: list of (key, PIL loader). Appends one JSON line per item."""
    import torch
    times, n_err = [], 0
    for i, (key, get_image) in enumerate(items, 1):
        rec = {"key": key}
        t = time.perf_counter()
        try:
            im = get_image()
            rec["probs"] = model.yes_no_probs(im, qs)
            rec["error"] = None
        except Exception as e:
            rec.update(probs=None, error=f"{type(e).__name__}: {e}")
            n_err += 1
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        rec["seconds"] = round(time.perf_counter() - t, 2)
        write_jsonl(out_path, rec)
        if not rec["error"]:
            times.append(rec["seconds"])
        msg = f"ERROR {rec['error']}" if rec["error"] else \
            f"min mass {min(v['mass_yes_no'] for v in rec['probs'].values()):.2f}"
        print(f"[{label} {i}/{len(items)}] {key} {rec['seconds']:.1f} s {msg}")
        if i % ETA_EVERY == 0 and times:
            m = sum(times) / len(times)
            print(f"    -- mean {m:.2f} s/image, ETA {datetime.timedelta(seconds=int(m * (len(items) - i)))}, errors {n_err}")
    return {"items": len(items), "ok": len(times), "errors": n_err,
            "mean_s": round(sum(times) / len(times), 2) if times else None}


def image_loader(path):
    from PIL import Image

    def get():
        with Image.open(path) as im:
            im.load()
            return im.copy()
    return get


def prior_items(n_noise):
    from PIL import Image
    items = [("black", lambda: Image.fromarray(np.zeros((512, 512), np.uint8), "L"))]
    for s in range(n_noise):
        items.append((f"noise_{s}", lambda s=s: Image.fromarray(
            np.random.default_rng(s).integers(0, 256, (512, 512), dtype=np.uint8), "L")))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke", "real", "prior"])
    ap.add_argument("--n", type=int, default=20, help="smoke: number of images")
    ap.add_argument("--priming_only", action="store_true",
                    help="smoke: only the guided/control questions + their plain ones (prompt wording check)")
    ap.add_argument("--n_noise", type=int, default=5, help="prior: number of noise images")
    ap.add_argument("--limit", type=int, default=None, help="real: first N manifest rows")
    ap.add_argument("--manifest", default=os.path.join(REPO, "manifests", "vindr.csv"))
    ap.add_argument("--out", default=RESULTS_INF)
    ap.add_argument("--weights", default=os.path.join(REPO, "weights", "CXR-LLAVA-v2"))
    args = ap.parse_args()
    qs = vc.questions()

    if args.cmd == "real":
        out_path = os.path.join(args.out, "vindr_yesno.jsonl")
        man = pd.read_csv(args.manifest)
        if args.limit:
            man = man.head(args.limit)
        done, n_err = read_done(out_path)
        todo = man[~man["key"].isin(done)]
        print(f"manifest {len(man)}  done {len(man) - len(todo)}  to do {len(todo)}"
              + (f" (incl. {n_err} earlier errors)" if n_err else ""))
        if todo.empty:
            print("nothing to do")
            return
        items = [(r["key"], image_loader(r["image_path"])) for _, r in todo.iterrows()]
    elif args.cmd == "prior":
        out_path = os.path.join(args.out, "vindr_yesno_prior.jsonl")
        if os.path.exists(out_path):
            os.remove(out_path)  # deterministic and small: always recomputed
        items = prior_items(args.n_noise)
    os.makedirs(args.out, exist_ok=True)

    if args.cmd == "smoke" and args.priming_only:
        qs = priming_questions(qs)
    model = load(args)
    show_tokens(model, qs)
    if args.cmd == "smoke":
        cmd_smoke(args, model, qs)
        return

    import torch
    info = {"run_id": datetime.datetime.now().isoformat(timespec="microseconds"), "task": f"yesno_{args.cmd}",
            "start": datetime.datetime.now().isoformat(timespec="seconds"), **git_info(),
            "weights": os.path.abspath(args.weights), "gpu": torch.cuda.get_device_name(0),
            "questions": qs, "scoring": "p_yes = P(yes variants)/(P(yes)+P(no)) at first answer token, no generation"}
    info_path = os.path.join(args.out, "run_info.json")
    save_run_info(info_path, info)
    info["summary"] = run_images(model, qs, items, out_path, args.cmd)
    info["end"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_run_info(info_path, info)
    print(f"done: {info['summary']} -> {out_path}")


if __name__ == "__main__":
    main()
