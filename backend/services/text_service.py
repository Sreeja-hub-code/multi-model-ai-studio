import requests

# Ollama local text generation (llama3)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"


def generate_text(prompt: str, system: str = None) -> str:
    """Generate text using Ollama (llama3) and return plain text.

    Keep return type as a plain string to avoid breaking existing Flask/UI code.
    `system` is appended to the prompt for compatibility with existing routes.
    """

    # Preserve existing route behavior: if prompt is empty, return empty string.
    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        return ""

    system_clean = (system or "").strip() if system else ""
    final_prompt = f"{system_clean}\n\n{prompt_clean}" if system_clean else prompt_clean

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": final_prompt,
                "stream": False,
            },
            timeout=120,
        )

        data = response.json()
        return (data.get("response", "") or "").strip()

    except Exception as e:
        # Preserve existing behavior: raise so the route converts it into JSON error.
        raise RuntimeError(str(e))

