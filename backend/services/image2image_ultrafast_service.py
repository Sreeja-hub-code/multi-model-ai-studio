import os
import time
import logging
from typing import Optional, Tuple

import torch
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

logger = logging.getLogger(__name__)
print("[LOCAL IMG2IMG] ultrafast service module imported")

# Model for ultra-fast CPU mode
MODEL_ID = os.getenv("IMG2IMG_FAST_MODEL_ID", "runwayml/stable-diffusion-v1-5")

# Singleton pipeline (lazy init)
_PIPELINE: Optional[StableDiffusionImg2ImgPipeline] = None


def _get_device() -> torch.device:
    return torch.device("cpu")


def _apply_cpu_threading() -> None:
    # Speed knobs for CPU
    try:
        threads = int(os.getenv("IMG2IMG_TORCH_NUM_THREADS", "4"))
        torch.set_num_threads(max(1, threads))
    except Exception:
        pass

    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass


def _maybe_resize_for_fast(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / float(max(w, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h), resample=Image.BICUBIC)


def _get_pipeline() -> StableDiffusionImg2ImgPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    _apply_cpu_threading()

    logger.info("[FAST IMG2IMG] loading model: %s", MODEL_ID)

    device = _get_device()

    # Force offline optionally
    offline_only = os.getenv("IMG2IMG_OFFLINE_ONLY", "0") == "1"
    if offline_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Load lightweight img2img pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # CPU only
    pipe = pipe.to(device)

    # Fast scheduler
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass

    # CPU optimizations
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    try:
        pipe.safety_checker = None
    except Exception:
        pass

    _PIPELINE = pipe
    logger.info("[FAST IMG2IMG] model ready")
    return _PIPELINE


def image_to_image(image_path: str, prompt: str, output_path: str) -> str:
    if not output_path:
        raise ValueError("Missing output path")

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("No transformation prompt provided")

    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with Image.open(image_path) as img:
        input_image = img.convert("RGB")

    # FAST MODE defaults
    width = int(os.getenv("IMG2IMG_FAST_WIDTH", "256"))
    height = int(os.getenv("IMG2IMG_FAST_HEIGHT", "256"))
    num_inference_steps = int(os.getenv("IMG2IMG_FAST_STEPS", "4"))
    guidance_scale = float(os.getenv("IMG2IMG_FAST_GUIDANCE_SCALE", "2.0"))
    strength = float(os.getenv("IMG2IMG_FAST_STRENGTH", "0.5"))

    # Resize (cheap) for faster inference
    input_image = input_image.resize((width, height), resample=Image.BICUBIC)

    pipe = _get_pipeline()

    print("[FAST IMG2IMG] started")
    t0 = time.time()

    # StableDiffusionImg2ImgPipeline expects strength for img2img
    result = pipe(
        prompt=prompt,
        image=input_image,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
    )

    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError("Pipeline returned no images")

    images[0].save(output_path)

    t1 = time.time()
    print("[FAST IMG2IMG] completed in", round(t1 - t0, 2), "sec")
    return output_path

