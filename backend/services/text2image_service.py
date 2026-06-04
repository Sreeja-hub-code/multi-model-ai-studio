import os
import re
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Tuple

import torch
from diffusers import AutoPipelineForText2Image


from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "runwayml/stable-diffusion-v1-5"

# One globally cached, CPU-only pipeline.
_pipeline = None
_pipeline_model: Optional[str] = None
_pipeline_lock = threading.Lock()
_pipeline_loading = False

# Background/thread executor to avoid stalling the Flask server thread.
_executor = ThreadPoolExecutor(max_workers=1)



def _get_device() -> str:
    # Requirement: CPU-only optimization.
    return "cpu"



def _parse_overlay_text(prompt: str) -> Tuple[str, Optional[str]]:
    """Extract explicit overlay text from the prompt.

    Supported syntaxes (case-insensitive):
      - add text: YOUR TEXT
      - overlay: YOUR TEXT
      - caption: YOUR TEXT

    Returns:
      cleaned_prompt, overlay_text_or_none
    """

    # Prefer capturing the shortest overlay text up to the next comma if present.
    # e.g. "..., add text: SPACE MISSION, more" -> SPACE MISSION
    # If it's at the end (no comma afterwards), capture till end.
    patterns = [
        r"(?:add\s*text|overlay|caption)\s*:\s*(?P<text>[^,]+?)\s*(?:,|$)",
    ]




    for pat in patterns:
        m = re.search(pat, prompt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        overlay_text = (m.group("text") or "").strip()
        if not overlay_text:
            return prompt, None

        # Remove only the overlay clause; keep the rest of the prompt.
        # Remove the overlay clause only if it's located near the end.
        # This avoids wiping earlier prompt fragments like:
        #   "Astronaut add text: SPACE MISSION, more"
        # which should keep "Astronaut".
        match = re.search(pat, prompt, flags=re.IGNORECASE | re.DOTALL)
        cleaned = prompt
        if match:
            start_idx = match.start()
            # treat as "near end" if it starts within the last ~30% of the string
            if start_idx >= int(len(prompt) * 0.7):
                cleaned = re.sub(pat, "", prompt, count=1, flags=re.IGNORECASE | re.DOTALL).strip()

        # Also remove any trailing commas left after clause removal.
        cleaned = re.sub(r"\s*,\s*$", "", cleaned).strip()


        return cleaned if cleaned else prompt, overlay_text

    return prompt, None


def _choose_font(font_size: int):
    """Best-effort font selection across environments."""

    candidates = [
        # Common paths on Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _overlay_centered_text(image: Image.Image, text: str) -> Image.Image:
    """Overlay centered caption text with white fill and dark outline/shadow."""

    if not text:
        return image

    img = image.convert("RGBA")
    draw = ImageDraw.Draw(img)

    w, h = img.size

    # Adaptive font sizing (clamp to keep it readable).
    # Start with ~7% of the shorter side.
    base = int(min(w, h) * 0.07)
    font_size = max(16, min(96, base))

    # Wrap text into multiple lines to fit.
    def wrap_lines(font: ImageFont.ImageFont) -> list[str]:
        # rough width-based wrapping
        words = re.split(r"\s+", text.strip())
        lines: list[str] = []
        cur: list[str] = []
        for word in words:
            test = " ".join(cur + [word]).strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            tw = bbox[2] - bbox[0]
            if tw <= w * 0.86:
                cur.append(word)
            else:
                if cur:
                    lines.append(" ".join(cur))
                cur = [word]
        if cur:
            lines.append(" ".join(cur))
        return lines

    # Slightly adjust font down if lines don't fit vertically.
    for _ in range(8):
        font = _choose_font(font_size)
        lines = wrap_lines(font)
        line_h = int(font_size * 1.15)
        total_h = line_h * len(lines)
        if total_h <= h * 0.25:
            break
        font_size = max(16, int(font_size * 0.9))

    font = _choose_font(font_size)
    lines = wrap_lines(font)
    line_h = int(font_size * 1.15)

    # Compute top-left for centered block
    block_w = 0
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        tw = bbox[2] - bbox[0]
        block_w = max(block_w, tw)
    block_h = line_h * len(lines)

    x0 = int((w - block_w) / 2)
    y0 = int((h - block_h) / 2)  # centered vertically

    # Draw shadow/outline
    for i, ln in enumerate(lines):
        y = y0 + i * line_h
        bbox = draw.textbbox((0, 0), ln, font=font)
        tw = bbox[2] - bbox[0]
        x = int((w - tw) / 2)

        # dark outline
        outline_w = max(2, int(font_size * 0.06))
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx * dx + dy * dy <= outline_w * outline_w:
                    draw.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0, 200))

        # white fill
        draw.text((x, y), ln, font=font, fill=(255, 255, 255, 255))

    return img


def _configure_pipeline_for_cpu(pipe):
    # Heavy CPU optimizations. Best-effort depending on diffusers version.
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # Some pipelines expose VAE slicing.
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    # Disable safety checker if present (Stable Diffusion v1.5).
    try:
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
    except Exception:
        pass

    return pipe


def _load_pipeline(timeout_sec: int = 300):
    """Load a single globally cached diffusers pipeline (CPU-only).

    Prevents repeated initialization and supports concurrent requests.
    """
    global _pipeline, _pipeline_model

    if _pipeline is not None:
        return _pipeline

    # Only one thread loads the model.
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        print(f"[text2image] model loading started model={MODEL_ID} device=cpu")
        start = time.time()

        hf_token = os.getenv("HF_TOKEN", "").strip() or None
        device = "cpu"

        try:
            # Ensure offline-capable behavior after first download:
            # - first attempt: allow download (default)
            # - if it fails (e.g., no internet), retry local-only
            for attempt, local_only in [(1, False), (2, True)]:
                try:
                    kwargs = dict(
                        torch_dtype=torch.float32,
                        device_map=None,
                        low_cpu_mem_usage=True,
                    )
                    if hf_token:
                        kwargs["token"] = hf_token

                    if "local_files_only" in AutoPipelineForText2Image.from_pretrained.__code__.co_varnames:
                        kwargs["local_files_only"] = local_only

                    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **kwargs)
                    pipe = pipe.to(device)
                    pipe = _configure_pipeline_for_cpu(pipe)

                    _pipeline = pipe
                    _pipeline_model = MODEL_ID

                    elapsed = time.time() - start
                    print(f"[text2image] model loading completed model={MODEL_ID} device=cpu time_sec={elapsed:.2f}")
                    return _pipeline
                except Exception as e:
                    last_err = e
                    print(f"[text2image] model loading attempt={attempt} local_only={local_only} failed: {e}")
                    if attempt == 2:
                        raise

        except Exception as e:
            raise RuntimeError(f"Failed to load pipeline {MODEL_ID}: {e}")



def _generate_image_sync(prompt: str, output_path: str, timeout_sec: int = 180) -> str:
    """Synchronous CPU-only generation with robust error handling."""

    # Validation (avoid diffusers choking on weird prompts).
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Invalid prompt: prompt must be a non-empty string")

    device = "cpu"
    pipe = _load_pipeline()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cleaned_prompt, overlay_text = _parse_overlay_text(prompt)

    gen_kwargs = {
        "prompt": cleaned_prompt,
        "num_inference_steps": 10,
        "guidance_scale": 5,
        "height": 384,
        "width": 384,
    }

    print(f"[text2image] generation started device={device} model={_pipeline_model}")
    print(f"[text2image] prompt='{prompt[:120]}'")
    print(f"[text2image] cleaned_prompt='{cleaned_prompt[:120]}'")
    if overlay_text:
        print(f"[text2image] overlay_applied=True text='{overlay_text[:120]}'")
    else:
        print("[text2image] overlay_applied=False")

    start = time.time()

    try:
        # Thread-safe guard for the global pipeline/model.
        with _pipeline_lock:
            result = pipe(**gen_kwargs)

        image: Image.Image = result.images[0]

        if overlay_text:
            image = _overlay_centered_text(image, overlay_text)

        image.save(output_path)

        elapsed = time.time() - start
        print(f"[text2image] generation completed saved={output_path} gen_time_sec={elapsed:.2f}")
        return output_path

    except FuturesTimeoutError:
        raise TimeoutError(f"Text-to-image generation timed out after {timeout_sec}s")

    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(f"Out of memory during generation (cuda). Details: {e}")

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "oom" in msg:
            raise RuntimeError(f"Out of memory during generation. Details: {e}")
        raise

    except Exception as e:
        raise RuntimeError(f"Text-to-image generation failed: {e}")


def generate_image(prompt: str, output_path: str) -> str:
    """Background-safe wrapper so Flask UI doesn't freeze.

    Uses a single worker executor and waits for the job with a timeout.
    This prevents global interpreter deadlocks and makes model init/generation
    safer under concurrent access.
    """

    timeout_sec = int(os.getenv("TEXT2IMAGE_TIMEOUT_SEC", "180"))

    fut = _executor.submit(_generate_image_sync, prompt, output_path, timeout_sec)
    try:
        return fut.result(timeout=timeout_sec + 5)
    except FuturesTimeoutError:
        raise TimeoutError(f"Text-to-image job timed out after {timeout_sec + 5}s")




