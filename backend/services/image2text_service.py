import base64
import os

import requests

# Ollama vision (llava)
OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"



def _guess_mode(mode: str) -> str:
    m = (mode or "describe").strip().lower()
    if m in {"describe", "analyze", "caption", "extract", "ocr"}:
        return m
    return "describe"


def _build_prompt(user_mode: str, user_hint: str = "") -> str:
    mode = _guess_mode(user_mode)

    # Keep this prompt short to reduce latency.
    if mode in {"describe", "analyze"}:
        base = "Describe the image in detail."
    elif mode == "caption":
        base = "Write a concise caption for the image."
    elif mode in {"extract", "ocr"}:
        base = "Extract all readable text from the image (OCR)."
    else:
        base = "Describe the image."

    if user_hint:
        return f"{base}\n\nFocus: {user_hint.strip()}"
    return base


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_text(image_path: str, mode: str = "describe") -> str:
    """Analyze image using local Ollama vision (llava) and return plain text."""

    # 1) Ensure uploaded image file exists before processing.
    if not image_path:
        raise FileNotFoundError("No image_path provided")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 2) Read image file and convert to base64.
    # (Requirement provides the canonical snippet; keep identical behavior.)
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 6) Support modes: describe, caption, OCR/text extraction, analyze.
    prompt = _build_prompt(mode)

    # 3) Send request to Ollama (streaming disabled for CPU-friendly, simpler parsing).
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            },
            timeout=120,
        )

        # 4) Parse Ollama JSON response correctly and return plain text.
        response.raise_for_status()
        data = response.json()

        text = (data.get("response") or "").strip()
        if text:
            return text

        # Some Ollama responses might use other keys; handle gracefully.
        for k in ("message", "output"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        raise ValueError(f"Invalid Ollama response payload: {data}")

    except requests.exceptions.Timeout as e:
        raise TimeoutError("Ollama request timed out") from e
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            "Failed to connect to Ollama. Is it running at http://localhost:11434 ?"
        ) from e
    except ValueError:
        # Re-raise invalid/unknown response format.
        raise
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Image analysis failed: {e}") from e


