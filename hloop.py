#!/usr/bin/env python3
"""
Description Diffusion Loop — describe an image with a vision LLM, rediffuse it
using that description as the prompt, repeat. Each cycle of lossy translation
(image → text → image) compounds reinterpretations, producing a perceptual
drift away from the original.

Supports CUDA (Linux/Windows) and MPS (Apple Silicon).
"""

import argparse
import base64
import gc
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    else:
        print("Warning: No GPU detected, falling back to CPU (very slow)", flush=True)
        return "cpu"

DEVICE = get_device()

# ---------------------------------------------------------------------------
# Vision LLM — image description via OpenAI-compatible API (LM Studio, etc.)
# ---------------------------------------------------------------------------

VISION_API_URL = os.environ.get("VISION_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
VISION_MODEL = os.environ.get("VISION_MODEL", "")

STARTERS = [
    "This image shows", "Looking at this image,", "The scene depicts",
    "Here we see", "What stands out is", "The photograph captures",
    "In this image,", "The composition features", "At first glance,",
    "The focal point is", "Dominating the frame is", "The image reveals",
    "Captured here is", "The artwork presents", "Visible in the frame is",
    "The picture displays", "Central to this image is", "Rendered in detail,",
    "The subject of this image is", "Spread across the frame,",
]

DESCRIBE_PROMPT = (
    'Describe only the visual appearance of this image in one line — '
    '(what does it look like, what it reminds). '
    'Do not identify the artwork, artist, or any context. Start with "{}"'
)


def describe_image(
    image_path: str,
    max_tokens: int = 150,
    temperature: float = 0.7,
    retries: int = 5,
) -> str:
    """Describe an image using a vision LLM via an OpenAI-compatible API."""
    prompt = DESCRIBE_PROMPT.format(random.choice(STARTERS))

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "webp": "image/webp", "gif": "image/gif",
    }.get(ext, "image/png")

    payload = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    if VISION_MODEL:
        payload["model"] = VISION_MODEL

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                VISION_API_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            wait = 10 * (attempt + 1)
            print(f"  [describe] attempt {attempt+1}/{retries} failed: {e}. "
                  f"Retrying in {wait}s...", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Vision API failed after {retries} retries")

# ---------------------------------------------------------------------------
# Color matching — prevents drift toward grey/brown
# ---------------------------------------------------------------------------

def match_color(image: Image.Image, reference: Image.Image, amount: float = 0.5) -> Image.Image:
    """Partially match per-channel mean/std to a reference image."""
    img = np.array(image, dtype=np.float32)
    ref = np.array(reference, dtype=np.float32)
    corrected = img.copy()
    for c in range(3):
        mu_img, std_img = img[:, :, c].mean(), img[:, :, c].std() + 1e-6
        mu_ref, std_ref = ref[:, :, c].mean(), ref[:, :, c].std() + 1e-6
        corrected[:, :, c] = (img[:, :, c] - mu_img) * (std_ref / std_img) + mu_ref
    blended = img * (1 - amount) + corrected * amount
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def flush():
    """Free GPU memory between steps."""
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str, dtype: str = "auto"):
    from diffusers import StableDiffusionXLImg2ImgPipeline

    # Determine dtype
    if dtype == "auto":
        torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    else:
        torch_dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype]

    print(f"Loading {model_id} ({torch_dtype})...", flush=True)
    t0 = time.time()

    kwargs = dict(torch_dtype=torch_dtype, use_safetensors=True)
    # Try fp16 variant first (some models only ship fp16 weights)
    try:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, variant="fp16", **kwargs)
    except OSError:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, **kwargs)

    pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    print(f"Ready ({time.time() - t0:.1f}s)", flush=True)
    return pipe

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

DEFAULT_W, DEFAULT_H = 1280, 720


def hallucination_loop(
    pipe,
    input_path: str,
    *,
    iterations: int = 300,
    strength: float = 0.3,
    guidance: float = 7.5,
    steps: int = 20,
    seed: Optional[int] = None,
    output_dir: str = "output",
    width: int = DEFAULT_W,
    height: int = DEFAULT_H,
    max_tokens: int = 150,
    temperature: float = 1.5,
    negative_prompt: Optional[str] = None,
):
    out = Path(output_dir)
    img_dir = out / "images"
    desc_dir = out / "descriptions"
    img_dir.mkdir(parents=True, exist_ok=True)
    desc_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume support ----
    existing = sorted(img_dir.glob("step_*.png"),
                      key=lambda f: int(f.stem.split("_")[1]))
    if existing:
        last_step = int(existing[-1].stem.split("_")[1])
        start_from = last_step + 1
        print(f"Resuming from step {start_from} "
              f"(found {len(existing)} existing images)", flush=True)
        image = Image.open(existing[-1]).convert("RGB")
        prev_image = image.copy()
        last_desc = desc_dir / f"step_{last_step:04d}.txt"
        if last_desc.exists():
            description = last_desc.read_text().strip()
        else:
            description = describe_image(
                str(existing[-1]), max_tokens=max_tokens, temperature=temperature)
        log = []
    else:
        start_from = 1
        image = Image.open(input_path).convert("RGB").resize(
            (width, height), Image.LANCZOS)
        prev_image = image.copy()

        step0 = img_dir / "step_0000.png"
        image.save(step0)
        print(f"Step 0000 (input) -> {step0}", flush=True)

        description = describe_image(
            str(step0), max_tokens=max_tokens, temperature=temperature)
        (desc_dir / "step_0000.txt").write_text(description)
        print(f"  Description: {description[:120]}...", flush=True)

        log = [{"step": 0, "image": str(step0), "description": description}]

    total_t0 = time.time()

    for i in range(start_from, iterations + 1):
        t0 = time.time()

        gen = torch.Generator(device=DEVICE)
        step_seed = (seed + i) if seed is not None else random.randint(0, 2**32 - 1)
        gen.manual_seed(step_seed)

        result = pipe(
            prompt=description,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        ).images[0]

        image = match_color(result, prev_image)

        out_path = img_dir / f"step_{i:04d}.png"
        image.save(out_path)

        description = describe_image(
            str(out_path), max_tokens=max_tokens, temperature=temperature)
        (desc_dir / f"step_{i:04d}.txt").write_text(description)

        elapsed = time.time() - t0
        print(f"Step {i:04d}/{iterations} ({elapsed:.1f}s) -> {out_path}", flush=True)
        print(f"  {description[:120]}...", flush=True)

        log.append({"step": i, "image": str(out_path), "description": description})
        prev_image = image.copy()
        flush()

    log_path = out / "log.json"
    log_path.write_text(json.dumps(log, indent=2))

    total = time.time() - total_t0
    print(f"\nDone! {iterations} steps in {total:.0f}s", flush=True)
    print(f"Output: {out.resolve()}", flush=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Description Diffusion Loop: describe -> rediffuse -> repeat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to the input image")
    p.add_argument("-n", "--iterations", type=int, default=300,
                   help="Number of loop iterations")
    p.add_argument("-s", "--strength", type=float, default=0.3,
                   help="img2img noise strength (0.0-1.0)")
    p.add_argument("-g", "--guidance", type=float, default=7.5,
                   help="CFG guidance scale")
    p.add_argument("--steps", type=int, default=20,
                   help="Denoising steps per iteration")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: random each step)")
    p.add_argument("-m", "--model", type=str,
                   default="RunDiffusion/Juggernaut-XL-v9",
                   help="HuggingFace SDXL model ID")
    p.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto",
                   help="Model precision (auto: fp16 on CUDA, fp32 on MPS)")
    p.add_argument("-o", "--output-dir", type=str, default="output",
                   help="Output directory")
    p.add_argument("-W", "--width", type=int, default=DEFAULT_W)
    p.add_argument("-H", "--height", type=int, default=DEFAULT_H)
    p.add_argument("--max-tokens", type=int, default=150,
                   help="Max tokens for image descriptions")
    p.add_argument("-t", "--temperature", type=float, default=1.5,
                   help="Vision LLM temperature (higher = more creative)")
    p.add_argument("--negative-prompt", type=str, default=None,
                   help="Negative prompt for diffusion")
    args = p.parse_args()

    pipe = load_pipeline(args.model, dtype=args.dtype)
    hallucination_loop(
        pipe,
        input_path=args.input,
        iterations=args.iterations,
        strength=args.strength,
        guidance=args.guidance,
        steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        negative_prompt=args.negative_prompt,
    )


if __name__ == "__main__":
    main()
