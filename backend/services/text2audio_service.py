import os


def text_to_audio(text: str, output_path: str, lang: str = "en") -> str:
    """Convert text to speech using gTTS."""
    try:
        from gtts import gTTS
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return output_path
    except ImportError:
        raise RuntimeError("gTTS not installed. Run: pip install gtts")
    except Exception as e:
        # Fallback to pyttsx3 (offline)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            return output_path
        except Exception as e2:
            raise RuntimeError(f"TTS failed: {e} | {e2}")
