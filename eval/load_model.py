"""Step 2: load CXR-LLaVA v2 in 4-bit on an 8 GB GPU (+ smoke test).

LLaMA layers -> 4-bit NF4 (bitsandbytes), compute in fp16.
Vision tower, mm_projector, lm_head, embeddings -> fp16 (not quantized).
fp16 instead of the paper's bf16 because Turing GPUs have no bf16.

The model class is imported from this repo's CXR_LLAVA_HF/ (not via
trust_remote_code), so edits in CXR_LLAVA_HF/CXR_LLAVA_HF.py are what runs.

Usage (on the server, from anywhere):
    python eval/load_model.py --images IMG/img.jpg a.png b.jpg ...
"""
import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from CXR_LLAVA_HF.CXR_LLAVA_HF import CXRLLAVAConfig, CXRLLAVAModel  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(REPO, "weights", "CXR-LLAVA-v2")
NOT_QUANTIZED = ["vision_tower", "mm_projector", "lm_head"]


def load_model(weights_dir=DEFAULT_WEIGHTS, device=0):
    from transformers import BitsAndBytesConfig

    weights_dir = os.path.abspath(weights_dir)
    config = CXRLLAVAConfig.from_pretrained(weights_dir)
    config.compute_dtype = "fp16"
    config._name_or_path = weights_dir  # tokenizer + generation_config are read from here

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        llm_int8_skip_modules=NOT_QUANTIZED,
    )
    model = CXRLLAVAModel.from_pretrained(
        weights_dir,
        config=config,
        torch_dtype=torch.float16,
        quantization_config=bnb,
        device_map={"": device},
        use_safetensors=True,
    )
    model.eval()
    return model


def check_model(model):
    """Print what got quantized and fail loudly if the split is not as intended."""
    import bitsandbytes as bnb

    def n4bit(module):
        return sum(isinstance(m, bnb.nn.Linear4bit) for m in module.modules())

    def dtypes(module):
        return sorted({str(p.dtype).replace("torch.", "") for p in module.parameters()})

    parts = {
        "llama": model.llama,
        "vision_tower": model.vision_tower,
        "mm_projector": model.mm_projector,
        "lm_head": model.lm_head,
    }
    print("\n== Model check ==")
    for name, mod in parts.items():
        print(f"  {name:13s} 4-bit layers: {n4bit(mod):4d}   param dtypes: {dtypes(mod)}")
    print(f"  embed_tokens  dtype: {model.llama.embed_tokens.weight.dtype}")
    print(f"  footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    assert n4bit(model.llama) > 0, "LLaMA was NOT quantized"
    for name in NOT_QUANTIZED:
        assert n4bit(parts[name]) == 0, f"{name} was quantized but should be fp16"
        assert dtypes(parts[name]) == ["float16"], f"{name} is not fp16: {dtypes(parts[name])}"


def check_vision_finite(model, image):
    """Run only the vision encoder + projector and check for NaN/inf (fp16 overflow)."""
    import numpy as np

    arr = np.expand_dims(np.array(image.convert("L")), -1)
    px = model.vision_tower.image_processor(arr, return_tensors="pt")["pixel_values"].to(model.device)
    with torch.no_grad():
        feats = model.encode_images(px)
    ok = bool(torch.isfinite(feats).all())
    print(f"  image features: shape {tuple(feats.shape)}, dtype {feats.dtype}, "
          f"finite={ok}, abs max={feats.abs().max().item():.1f}")
    return ok


def main():
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--images", nargs="+", default=[os.path.join(REPO, "IMG", "img.jpg")])
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (as in the plan)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    t0 = time.perf_counter()
    model = load_model(args.weights)
    print(f"Loaded in {time.perf_counter() - t0:.0f} s, "
          f"VRAM allocated after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    check_model(model)

    print("\n== Vision fp16 check ==")
    if not check_vision_finite(model, Image.open(args.images[0])):
        print("!! NaN/inf in image features -> fp16 overflow; vision tower needs fp32. Stopping.")
        sys.exit(1)

    print("\n== Reports ==")
    torch.cuda.reset_peak_memory_stats()
    times = []
    for path in args.images:
        image = Image.open(path)
        torch.cuda.synchronize()
        t = time.perf_counter()
        report = model.generate_report(image, temperature=args.temperature)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        times.append(dt)
        n_tok = len(model.tokenizer(report).input_ids)
        print(f"\n--- {path}  ({image.mode} {image.size[0]}x{image.size[1]})")
        print(f"    {dt:.1f} s, {n_tok} tokens, {dt / max(n_tok, 1):.3f} s/token")
        print(report if report else "!! EMPTY REPORT")

    print("\n== Summary ==")
    print(f"  peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  peak VRAM reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    print(f"  s/image: mean {sum(times) / len(times):.1f}"
          + (f", mean excl. first (warm-up) {sum(times[1:]) / len(times[1:]):.1f}" if len(times) > 1 else ""))


if __name__ == "__main__":
    main()
