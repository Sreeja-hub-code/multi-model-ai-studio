"""backend.services.audio2text_service

Audio -> Text services.

This module supports:
1) upload-based transcription via `audio_to_text(audio_path)` (backwards compatible)
2) realtime microphone transcription via a streaming-safe engine:
   - `RealtimeTranscriber.start()` / `RealtimeTranscriber.stop()`
   - `RealtimeTranscriber.transcript_stream()` yields partial updates

Design goals:
- CPU-only (faster-whisper)
- low latency chunk-based processing
- avoid blocking Flask requests (background thread + queue)
- CPU/RAM-friendly for low-end laptops

Note:
- Microphone capture uses `sounddevice` and requires OS microphone permission.
- faster-whisper, sounddevice, numpy must be installed.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import logging
from dataclasses import dataclass
from typing import Generator, Optional, List, Tuple

logger = logging.getLogger(__name__)


# -----------------------------
# Dependency / environment checks
# -----------------------------

def _require_deps() -> None:
    """Raise a helpful error if required deps are missing."""

    missing: List[str] = []
    for mod in ("faster_whisper", "sounddevice", "numpy"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)

    if missing:
        raise RuntimeError(
            "Missing required dependencies for realtime transcription: "
            + ", ".join(missing)
            + "\nInstall with:\n"
            + "  pip install faster-whisper sounddevice numpy\n"
        )


# -----------------------------
# Upload-based transcription (backwards compatible)
# -----------------------------


def audio_to_text(audio_path: str) -> str:
    """Convert an uploaded audio file into text.

    Backwards compatible API used by existing Flask route `/api/audio-to-text`.
    """

    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _require_deps()

    # Use faster-whisper for both file + streaming to keep model consistent.
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "faster-whisper is required. Install with: pip install faster-whisper\n"
            f"Error: {e}"
        )

    # Keep model loading lazy + cached.
    model = _get_whisper_model()
    segments, _info = model.transcribe(
        audio_path,
        vad_filter=True,
        # CPU-friendly defaults
        beam_size=1,
        language=None,
    )

    parts: List[str] = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if txt:
            parts.append(txt)

    return " ".join(parts).strip()


_whisper_model_lock = threading.Lock()
_whisper_model: Optional["WhisperModel"] = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    with _whisper_model_lock:
        if _whisper_model is not None:
            return _whisper_model

        from faster_whisper import WhisperModel

        # Model size: base by default (speed/accuracy). tiny also works but base
        # is often acceptable on CPU.
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")

        _whisper_model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",  # CPU-friendly
            cpu_threads=max(1, int(os.getenv("WHISPER_CPU_THREADS", "4"))),
        )
        return _whisper_model


# -----------------------------
# Realtime microphone transcription
# -----------------------------


@dataclass
class RealtimeConfig:
    sample_rate: int = 16000
    channels: int = 1

    # Chunking strategy
    # - We capture in small blocks via sounddevice callback.
    # - We then assemble into longer "windows" for faster-whisper.
    window_seconds: float = 2.0  # transcription window size
    step_seconds: float = 0.75  # how often to emit partial updates

    # VAD / silence handling (simple CPU-friendly)
    min_rms: float = 0.01  # below this is treated as near-silence
    max_silence_seconds: float = 2.5

    # Session limits
    max_duration_seconds: float = 60 * 5  # hard stop after 5 minutes
    max_queue_frames: int = 250  # avoid unbounded memory

    # faster-whisper options
    model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    beam_size: int = 1
    vad_filter: bool = False  # we do our own lightweight silence detection
    language: Optional[str] = None


class RealtimeTranscriber:
    """Realtime microphone transcription engine.

    Usage pattern:
        rt = RealtimeTranscriber()
        rt.start()
        for event in rt.transcript_stream():
            print(event["text"])
        rt.stop()

    `transcript_stream()` yields dict events:
        {"type":"partial","text":...,"final":False}
        {"type":"final","text":...,"final":True}
        {"type":"error","error":...}
    """

    def __init__(self, config: Optional[RealtimeConfig] = None):
        _require_deps()
        self.config = config or RealtimeConfig()

        import numpy as np

        self._np = np

        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=self.config.max_queue_frames
        )
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._transcribe_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._last_emitted_text = ""
        self._final_text = ""

        # sounddevice handle
        self._sd = None
        self._stream = None

        self._model = _get_whisper_model()

        # timing/state
        self._session_start: Optional[float] = None
        self._last_audio_ts: Optional[float] = None
        self._last_partial_ts: Optional[float] = None

    # -----------------------------
    # Device detection helpers
    # -----------------------------

    @staticmethod
    def list_input_devices() -> List[dict]:
        """Return available input devices for sounddevice."""
        try:
            import sounddevice as sd
        except Exception as e:
            raise RuntimeError(f"sounddevice missing: {e}")

        devices = []
        try:
            for idx, dev in enumerate(sd.query_devices() or []):
                # dev is dict
                pass
        except Exception:
            # sd.query_devices() already returns mapping/iterable.
            pass

        try:
            devs = sd.query_devices()
            if isinstance(devs, dict):
                # unlikely
                iterable = devs.items()
            else:
                iterable = enumerate(devs)
        except Exception:
            iterable = []

        # Use sd.query_devices() correctly
        try:
            devs = sd.query_devices()
            if hasattr(devs, "__iter__"):
                for idx, d in enumerate(devs):
                    if d is None:
                        continue
                    if d.get("max_input_channels", 0) > 0:
                        devices.append(
                            {
                                "index": idx,
                                "name": d.get("name"),
                                "max_input_channels": d.get("max_input_channels"),
                                "default_samplerate": d.get("default_samplerate"),
                            }
                        )
        except Exception:
            # fallback: ignore
            pass

        return devices

    # -----------------------------
    # Start/Stop
    # -----------------------------

    def start(self, device: Optional[int] = None) -> None:
        """Start capturing audio and transcription."""

        if self._started_event.is_set():
            return

        self._stop_event.clear()
        self._last_emitted_text = ""
        self._final_text = ""

        self._session_start = time.time()
        self._last_audio_ts = time.time()
        self._last_partial_ts = 0.0

        # Start transcription thread (consumes queue)
        self._transcribe_thread = threading.Thread(
            target=self._transcribe_loop,
            name="realtime-transcriber",
            daemon=True,
        )
        self._transcribe_thread.start()

        # Start capture (sounddevice stream) in current thread? No: in another thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(device,),
            name="realtime-capture",
            daemon=True,
        )
        self._capture_thread.start()

        self._started_event.set()

    def stop(self) -> None:
        """Stop session and release microphone."""
        self._stop_event.set()

        # Stop sounddevice stream
        try:
            if self._stream is not None:
                self._stream.stop(ignore_errors=True)
                self._stream.close(ignore_errors=True)
        except Exception:
            pass

    # -----------------------------
    # Streaming-safe generator
    # -----------------------------

    def transcript_stream(self) -> Generator[dict, None, None]:
        """Yield partial and final transcription events.

        Stability requirements for SSE:
        - Never block indefinitely.
        - Continuously check stop/fatal and keep generator alive.
        - Only emit final/error when transcription completes or fatal error occurs.
        """

        # Use a separate queue for events from transcribe thread.
        event_queue: "queue.Queue[dict]" = queue.Queue(maxsize=50)

        def _event_put(evt: dict) -> None:
            try:
                event_queue.put(evt, timeout=1)
            except queue.Full:
                # If frontend is slow, drop partials.
                pass

        # Monkey-patch callback via instance attribute.
        self._emit_event = _event_put  # type: ignore[attr-defined]

        # If the consumer is waiting for the first transcript while the engine
        # is still warming up, periodically wake up so the SSE route can send heartbeats.
        while not self._stop_event.is_set():
            try:
                evt = event_queue.get(timeout=0.5)
                yield evt
                if evt.get("type") in ("final", "error"):
                    return
            except queue.Empty:
                # No event yet; keep generator alive.
                continue
            except GeneratorExit:
                # Consumer disconnected.
                return

        # stop_event is set => only emit final as a best-effort if nothing was produced.
        if not self._final_text.strip():
            yield {"type": "final", "text": "", "final": True}

        return


    # -----------------------------
    # Capture loop
    # -----------------------------

    def _capture_loop(self, device: Optional[int]) -> None:
        import sounddevice as sd
        np = self._np

        # Device detection + permission handling
        try:
            input_devices = sd.query_devices()
        except Exception as e:
            err = (
                "Microphone access/device query failed. "
                "Ensure microphone permissions are granted. "
                f"Error: {e}"
            )
            self._emit_event({"type": "error", "error": err, "final": True})  # type: ignore[attr-defined]
            return

        chosen_device = device
        try:
            if chosen_device is None:
                # Prefer default input device
                default_in = sd.default.device[0]
                if default_in is not None:
                    chosen_device = int(default_in)
                else:
                    chosen_device = None

            # Validate device has input channels
            if chosen_device is not None:
                dev = input_devices[chosen_device] if isinstance(input_devices, (list, tuple)) else input_devices
                # If query_devices returns a list, dev is dict.
                if isinstance(input_devices, (list, tuple)) and dev is not None:
                    if dev.get("max_input_channels", 0) <= 0:
                        chosen_device = None
        except Exception:
            chosen_device = None

        logger.info("[audio2text] recording started (device=%s)", chosen_device)

        # sounddevice callback pushes small blocks into queue
        # We request int16 PCM for cheap processing.
        blocksize = int(self.config.step_seconds * self.config.sample_rate)
        # Avoid too-large blocks; ensure at least ~10ms.
        blocksize = max(int(self.config.sample_rate * 0.01), min(blocksize, int(self.config.sample_rate * 0.5)))

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("[audio2text] sounddevice status: %s", status)

            if self._stop_event.is_set():
                return

            # indata shape: (frames, channels)
            audio = indata
            if audio is None:
                return
            # Convert to mono
            if audio.ndim == 2 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            elif audio.ndim == 2:
                audio = audio[:, 0]

            # Normalize to float32 in [-1,1]
            # indata is usually float32 already; but handle int16 just in case.
            if audio.dtype.kind in ("i", "u"):
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

            # queue audio block
            try:
                self._audio_queue.put_nowait(audio)
            except queue.Full:
                # drop blocks under pressure
                pass

        try:
            self._sd = sd
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                device=chosen_device,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            )

            with self._stream:
                self._started_event.wait(timeout=1)
                while not self._stop_event.is_set():
                    time.sleep(0.05)

        except Exception as e:
            err = (
                "Microphone capture failed. "
                "Check permissions and input device availability. "
                f"Error: {e}"
            )
            logger.exception("[audio2text] capture error")
            self._emit_event({"type": "error", "error": err, "final": True})  # type: ignore[attr-defined]
            self._stop_event.set()
            return

    # -----------------------------
    # Transcribe loop (consumes queue)
    # -----------------------------

    def _transcribe_loop(self) -> None:
        import numpy as np

        logger.info("[audio2text] transcription started")

        # Rolling buffer of audio samples (float32)
        window_samples = int(self.config.window_seconds * self.config.sample_rate)
        step_samples = max(1, int(self.config.step_seconds * self.config.sample_rate))

        buffer = np.zeros((0,), dtype=np.float32)
        last_partial_text = ""
        final_text_parts: List[str] = []

        try:
            while not self._stop_event.is_set():
                # Timeout waiting for audio
                try:
                    block = self._audio_queue.get(timeout=0.5)
                    if block is None:
                        continue
                except queue.Empty:
                    # Timeout => if user stops, break.
                    # Also handle max silence.
                    if time.time() - (self._last_audio_ts or time.time()) > self.config.max_silence_seconds:
                        break
                    continue

                # Update silence tracking
                rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
                if rms >= self.config.min_rms:
                    self._last_audio_ts = time.time()

                # Append to buffer
                if buffer.size == 0:
                    buffer = block.copy()
                else:
                    buffer = np.concatenate([buffer, block])

                # Cap buffer to avoid RAM growth
                if buffer.size > window_samples:
                    buffer = buffer[-window_samples:]

                # Emit partial when we have enough samples and step interval elapsed
                now = time.time()
                if buffer.size < int(self.config.window_seconds * self.config.sample_rate * 0.6):
                    continue

                if self._last_partial_ts is not None and now - self._last_partial_ts < self.config.step_seconds * 0.9:
                    continue

                self._last_partial_ts = now

                # CPU-friendly: transcribe only the last window
                audio_window = buffer.astype(np.float32)

                # faster-whisper expects int16 or float32? We'll pass float32 and let it handle.
                segments, _info = self._model.transcribe(
                    audio_window,
                    language=self.config.language,
                    beam_size=self.config.beam_size,
                    vad_filter=self.config.vad_filter,
                )

                texts: List[str] = []
                for seg in segments:
                    t = (seg.text or "").strip()
                    if t:
                        texts.append(t)

                candidate = " ".join(texts).strip()
                if not candidate:
                    continue

                # De-dup / incremental emission: if candidate starts with last emitted, keep candidate,
                # else try to avoid repeating tail.
                with self._lock:
                    new_text = candidate
                    if last_partial_text and new_text.startswith(last_partial_text):
                        # ok
                        pass
                    else:
                        # naive tail de-dup
                        # find longest overlap between last_partial_text suffix and candidate prefix
                        overlap = ""
                        for k in range(1, min(12, len(last_partial_text.split()) + 1)):
                            last_tail = " ".join(last_partial_text.split()[-k:])
                            if last_tail and new_text.startswith(last_tail):
                                overlap = last_tail
                        if overlap:
                            # still keep full candidate; only helps log clarity
                            pass

                    last_partial_text = new_text
                    self._last_emitted_text = new_text

                logger.info("[audio2text] partial text received: %s", new_text)

                self._emit_event({"type": "partial", "text": new_text, "final": False})  # type: ignore[attr-defined]

                # Heuristic finalization: if silence exceeded, break in outer logic.
                # Also hard-stop by duration.
                if self._session_start and (time.time() - self._session_start) > self.config.max_duration_seconds:
                    break

            # Finalize
            final_text = self._last_emitted_text.strip()
            if final_text:
                final_text_parts = [final_text]
            completed = " ".join(final_text_parts).strip() if final_text_parts else final_text

            self._final_text = completed
            logger.info("[audio2text] transcription completed")
            self._emit_event({"type": "final", "text": completed, "final": True})  # type: ignore[attr-defined]

        except Exception as e:
            logger.exception("[audio2text] transcription error")
            self._emit_event({"type": "error", "error": str(e), "final": True})  # type: ignore[attr-defined]
            self._stop_event.set()

