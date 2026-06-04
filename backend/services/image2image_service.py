import logging
import os
from typing import Optional

import torch
from PIL import Image

from diffusers import StableDiffusionInstructPix2PixPipeline

logger = logging.getLogger(__name__)
print("[LOCAL IMG2IMG] image2image_service module imported")

# --- Model configuration (local, no HuggingFace API URLs) -----------------
MODEL_ID = "timbrooks/instruct-pix2pix"
# If you need fully offline execution, ensure the model is already present in the local diffusers/huggingface cache.
# Otherwise diffusers will attempt to reach the HuggingFace Hub.
OFFLINE_ONLY = os.getenv("IMG2IMG_OFFLINE_ONLY", "0") == "1"

# Singleton pipeline (lazy init)
_PIPELINE: Optional[StableDiffusionInstructPix2PixPipeline] = None


def _get_device() -> torch.device:
    # Requirement: CPU-only inference
    return torch.device("cpu")


def _get_pipeline() -> StableDiffusionInstructPix2PixPipeline:
    """Load the instruct-pix2pix pipeline once and reuse."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    logger.info("[image] loading local instruct-pix2pix model")

    device = _get_device()

    try:
        # CPU optimizations / lower memory usage
        # Force offline if requested; prevents any network calls to the Hub.
        if OFFLINE_ONLY:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        _PIPELINE = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        # Ensure CPU execution
        _PIPELINE = _PIPELINE.to(device)

        # ---- CPU optimizations ----
        # These are safe/no-op when the underlying diffusers version doesn't support them.
        try:
            _PIPELINE.enable_attention_slicing()
        except Exception:
            pass

        try:
            _PIPELINE.enable_vae_slicing()
        except Exception:
            pass

        # CPU offload can help memory; keep it optional.
        try:
            if hasattr(_PIPELINE, "enable_model_cpu_offload"):
                _PIPELINE.enable_model_cpu_offload()
        except Exception:
            pass

        # Disable safety checker if present (speed)
        try:
            if hasattr(_PIPELINE, "safety_checker"):
                _PIPELINE.safety_checker = None
        except Exception:
            pass


        logger.info("[image] model ready")
        return _PIPELINE

    except Exception as e:
        # Make failures explicit
        _PIPELINE = None
        logger.exception("[image] model load failure")
        raise RuntimeError(f"Failed to load local instruct-pix2pix model: {e}")


def _validate_image(image_path: str) -> None:
    if not image_path:
        raise ValueError("Missing image path")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")


def image_to_image(image_path: str, prompt: str, output_path: str) -> str:
    """Transform image using local Diffusers instruct-pix2pix."""

    if not output_path:
        raise ValueError("Missing output path")

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("No transformation prompt provided")

    _validate_image(image_path)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Load and validate input image
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            input_image = img
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")

    pipeline = _get_pipeline()

    logger.info("[image] transformation started")

    try:
        # ---- FAST CPU defaults (tuned for speed) ----
        # Reduce steps/resolution + relax guidance for faster inference.
        # You can override via env vars without changing code.
        fast_max_side = int(os.getenv("IMG2IMG_FAST_MAX_SIDE", "384"))
        num_steps = int(os.getenv("IMG2IMG_NUM_STEPS", "8"))  # 6–10
        image_guidance_scale = float(os.getenv("IMG2IMG_IMAGE_GUIDANCE_SCALE", "1.0"))
        guidance_scale = float(os.getenv("IMG2IMG_GUIDANCE_SCALE", "3.5"))

        # Resize input to reduce compute
        w, h = input_image.size
        scale = min(1.0, fast_max_side / float(max(w, h)))
        if scale != 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            input_image = input_image.resize((new_w, new_h), resample=Image.BICUBIC)

        print("[image] generation started")
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        # StableDiffusionInstructPix2PixPipeline call
        result = pipeline(
            prompt=prompt,
            image=input_image,
            num_inference_steps=num_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
        )

        images = getattr(result, "images", None)
        if not images:
            raise RuntimeError("Pipeline returned no images")

        out_img = images[0]
        out_img.save(output_path)

        logger.info("[image] transformation completed")
        return output_path

    except Exception as e:
        logger.exception("[image] inference error")
        raise RuntimeError(f"Image transformation failed: {e}")

