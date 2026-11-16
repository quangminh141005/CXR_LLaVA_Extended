"""Step 3.2: visual check of what the model actually sees.

For 3 images per dataset, applies the model's own preprocessing (the same
CLIPImageProcessor settings as CXR_LLAVA_HF.py, read from the weights' config.json)
and saves a side-by-side PNG:

    left  = original image, red box = the area kept by the 512x512 center crop
    right = the 512x512 model input (normalisation undone, so it looks like an X-ray)

No model / GPU needed. Paths are built from the dataset CSVs (no folder listing,
which hangs on /storage). Nothing is modified except the output folder.

Usage (on the server):
    python -u eval/check_preprocessing.py --out log/preprocess_check

Then look at log/preprocess_check/*.png (or overview.png). Check for:
inverted images (bones dark), black/blank images, rotated images, and whether
the crop cuts off lung apices or costophrenic angles.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from check_data import DEFAULTS  # noqa: E402  (same dataset paths as Step 0)

MIMIC_JPG_ROOT = "/storage/shared/MIMIC-CXR-JPG/split-4/files"  # complete copy (Step 0)
PANEL = 512


def build_processor(weights_dir):
    """Same arguments as CXRLLAVAModel.__init__ in CXR_LLAVA_HF.py."""
    import transformers

    with open(os.path.join(weights_dir, "config.json")) as f:
        cfg = json.load(f)
    size = cfg["clip_vision_cfg"]["image_size"]
    mean, std = cfg["image_preprocess_cfg"]["mean"], cfg["image_preprocess_cfg"]["std"]
    proc = transformers.CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": size},
        resample=True,
        do_center_crop=True,
        crop_size=size,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=mean,
        image_std=std,
        do_convert_rgb=False,
    )
    return proc, size, mean, std


def sample_paths(n, seed):
    """n image paths per dataset, built from the CSVs only."""
    d = DEFAULTS
    out = {}

    lab = pd.read_csv(d["vindr_labels"])
    ids = pd.Series(lab["image_id"].drop_duplicates()).sample(n, random_state=seed)
    out["VinDr"] = [os.path.join(d["vindr_png_dir"], f"{i}.png") for i in ids]

    gt = pd.read_csv(os.path.join(d["chexpert_root"], "groundtruth.csv"))
    keys = gt["Study"].str.extract(r"(patient\d+/study\d+)")[0].dropna().sample(n, random_state=seed)
    out["CheXpert"] = [os.path.join(d["chexpert_root"], "CheXpert", "test", "test", k, "view1_frontal.jpg")
                       for k in keys]

    proj = pd.read_csv(os.path.join(d["indiana_root"], "indiana_projections.csv"))
    fr = proj[proj["projection"].str.lower() == "frontal"]["filename"].sample(n, random_state=seed)
    out["NLMCXR"] = [os.path.join(d["indiana_root"], "images", "images_normalized", f) for f in fr]

    rows = pd.read_csv(d["mimic_csv"], header=None)[0].sample(n, random_state=seed)
    rels = [r.strip().replace("\\", "/").removeprefix("files/").replace(".dcm", ".jpg") for r in rows]
    out["MIMIC"] = [os.path.join(MIMIC_JPG_ROOT, r) for r in rels]
    return out


def crop_box(w, h):
    """Region of the original kept by 'resize shortest edge + center crop' (square of side min(w,h))."""
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return left, top, left + s, top + s


def model_input(proc, image, mean, std):
    """Exactly what generate_report() feeds the vision tower, converted back to a viewable uint8 image."""
    arr = np.expand_dims(np.array(image.convert("L")), -1)
    px = proc(arr, return_tensors="np")["pixel_values"][0]  # (1, 512, 512), normalised
    m, s = float(np.ravel(mean)[0]), float(np.ravel(std)[0])
    x = px[0] * s + m
    return Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8)), px


def side_by_side(image, inp):
    w, h = image.size
    scale = PANEL / max(w, h)
    left = image.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))))
    box = [round(v * scale) for v in crop_box(w, h)]
    ImageDraw.Draw(left).rectangle(box, outline=(255, 0, 0), width=3)
    canvas = Image.new("RGB", (PANEL * 2 + 10, PANEL), (40, 40, 40))
    canvas.paste(left, ((PANEL - left.width) // 2, (PANEL - left.height) // 2))
    canvas.paste(inp.convert("RGB"), (PANEL + 10, 0))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.path.join(REPO, "weights", "CXR-LLAVA-v2"))
    ap.add_argument("--out", default=os.path.join(REPO, "log", "preprocess_check"))
    ap.add_argument("--n", type=int, default=3, help="images per dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    proc, size, mean, std = build_processor(args.weights)
    print(f"processor: shortest_edge={size}, center crop {size}x{size}, mean={mean}, std={std}\n")

    rows = []
    for ds, paths in sample_paths(args.n, args.seed).items():
        for i, path in enumerate(paths):
            tag = f"{ds}_{i}"
            try:
                image = Image.open(path)
                image.load()
            except Exception as e:  # missing / unreadable -> report, keep going
                print(f"[{tag}] FAIL {path}: {e}")
                continue
            raw = np.array(image)
            w, h = image.size
            l, t, r, b = crop_box(w, h)
            cut = 1 - (r - l) * (b - t) / (w * h)
            note = "16-bit -> model REFUSES" if raw.max() > 255 else ""
            inp, px = model_input(proc, image, mean, std)
            panel = side_by_side(image, inp)
            panel.save(os.path.join(args.out, f"{tag}.png"))
            rows.append((tag, panel))
            print(f"[{tag}] {image.mode} {w}x{h} raw[{raw.min()}..{raw.max()}] mean {raw.mean():.0f} | "
                  f"crop removes {cut:.0%} ({'top/bottom' if h > w else 'left/right' if w > h else 'none'}) | "
                  f"input {tuple(px.shape)} [{px.min():.2f}..{px.max():.2f}] {note}\n        {path}")

    if rows:  # one overview sheet: one row per image
        sheet = Image.new("RGB", (rows[0][1].width, rows[0][1].height * len(rows)), (0, 0, 0))
        for k, (_, im) in enumerate(rows):
            sheet.paste(im, (0, k * im.height))
        sheet = sheet.resize((sheet.width // 2, sheet.height // 2))
        sheet.save(os.path.join(args.out, "overview.png"))
    print(f"\nsaved {len(rows)} images + overview.png to {args.out}")


if __name__ == "__main__":
    main()
