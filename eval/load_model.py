"""Step 2: load CXR-LLaVA v2 in 4-bit on an 8 GB GPU (+ smoke test).

LLaMA layers -> 4-bit NF4 (bitsandbytes), compute in fp16.
Vision tower, mm_projector, lm_head, embeddings -> fp16 (not quantized).
fp16 instead of the paper's bf16 because Turing GPUs have no bf16.

The model class is imported from this repo's CXR_LLAVA_HF/ (not via
trust_remote_code), so edits in CXR_LLAVA_HF/CXR_LLAVA_HF.py are what runs.

Step 2.5 adds a full-precision CPU reference (no quantization) to compare against:
    --device cpu --dtype fp32|bf16   (whole model in RAM, ~28 / ~14 GB)

Usage (on the server, from anywhere):
    python eval/load_model.py --images IMG/img.jpg a.png b.jpg ...                     # 4-bit GPU
    numactl --cpunodebind=0 --membind=0 \
      python eval/load_model.py --device cpu --dtype fp32 --threads 20 --images ...     # CPU reference
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


CPU_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load_model_cpu(weights_dir=DEFAULT_WEIGHTS, dtype="fp32"):
    """Full-precision reference: no quantization, whole model on CPU in fp32 or bf16 (authors' dtype)."""
    weights_dir = os.path.abspath(weights_dir)
    config = CXRLLAVAConfig.from_pretrained(weights_dir)
    config.compute_dtype = dtype
    config._name_or_path = weights_dir
    model = CXRLLAVAModel.from_pretrained(
        weights_dir,
        config=config,
        torch_dtype=CPU_DTYPES[dtype],
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.eval()
    return model


def check_model_cpu(model, dtype):
    """Reference must be unquantized and entirely in the requested dtype."""
    want = str(CPU_DTYPES[dtype]).replace("torch.", "")
    found = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
    devices = sorted({str(p.device) for p in model.parameters()})
    n_params = sum(p.numel() for p in model.parameters())
    print("\n== Model check (CPU reference) ==")
    print(f"  param dtypes: {found}   devices: {devices}   params: {n_params / 1e9:.2f} B")
    print(f"  footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    assert found == [want], f"expected only {want}, found {found}"
    assert devices == ["cpu"], f"expected cpu only, found {devices}"


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KB -> GB


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
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="cuda = 4-bit NF4 on GPU (default); cpu = unquantized reference")
    ap.add_argument("--dtype", choices=list(CPU_DTYPES), default="fp32", help="CPU reference dtype")
    ap.add_argument("--threads", type=int, default=20, help="CPU threads (20 = one socket)")
    args = ap.parse_args()
    gpu = args.device == "cuda"

    if gpu:
        assert torch.cuda.is_available(), "CUDA not available"
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        torch.set_num_threads(args.threads)
        print(f"CPU reference: dtype={args.dtype}, torch threads={torch.get_num_threads()}, "
              f"cores visible={len(os.sched_getaffinity(0))}")

    t0 = time.perf_counter()
    if gpu:
        model = load_model(args.weights)
        print(f"Loaded in {time.perf_counter() - t0:.0f} s, "
              f"VRAM allocated after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        check_model(model)
    else:
        model = load_model_cpu(args.weights, args.dtype)
        print(f"Loaded in {time.perf_counter() - t0:.0f} s, peak RAM so far: {peak_rss_gb():.1f} GB")
        check_model_cpu(model, args.dtype)

    print("\n== Vision fp16 check ==")
    if not check_vision_finite(model, Image.open(args.images[0])):
        print("!! NaN/inf in image features -> fp16 overflow; vision tower needs fp32. Stopping.")
        sys.exit(1)

    print("\n== Reports ==")
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for path in args.images:
        image = Image.open(path)
        if gpu:
            torch.cuda.synchronize()
        t = time.perf_counter()
        report = model.generate_report(image, temperature=args.temperature)
        if gpu:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        times.append(dt)
        n_tok = len(model.tokenizer(report).input_ids)
        print(f"\n--- {path}  ({image.mode} {image.size[0]}x{image.size[1]})")
        print(f"    {dt:.1f} s, {n_tok} tokens, {dt / max(n_tok, 1):.3f} s/token")
        print(report if report else "!! EMPTY REPORT")

    print("\n== Summary ==")
    if gpu:
        print(f"  peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"  peak VRAM reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    else:
        print(f"  mode: CPU {args.dtype}, {torch.get_num_threads()} threads, peak RAM {peak_rss_gb():.1f} GB")
    print(f"  s/image: mean {sum(times) / len(times):.1f}"
          + (f", mean excl. first (warm-up) {sum(times[1:]) / len(times[1:]):.1f}" if len(times) > 1 else ""))


if __name__ == "__main__":
    main()
