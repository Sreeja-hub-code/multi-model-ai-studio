import os
import uuid
from flask import Blueprint, request, jsonify, render_template, send_file, current_app, Response
from werkzeug.utils import secure_filename
from backend.services.audio2text_service import audio_to_text, RealtimeTranscriber

import threading


# -----------------------------
# LIVE transcription (SSE) sessions
# -----------------------------
# Keeps active RealtimeTranscriber instances in memory only.
# Each session is stopped on explicit stop or client disconnect.
_active_realtime_sessions = {}
_active_realtime_sessions_lock = threading.Lock()


def _create_session(session_id: str) -> RealtimeTranscriber:
    rt = RealtimeTranscriber()
    with _active_realtime_sessions_lock:
        # Stop any older session with same id (shouldn't normally happen, but be safe)
        old = _active_realtime_sessions.get(session_id)
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
        _active_realtime_sessions[session_id] = rt
    return rt


def _get_session(session_id: str) -> RealtimeTranscriber | None:
    with _active_realtime_sessions_lock:
        return _active_realtime_sessions.get(session_id)


def _remove_session(session_id: str) -> None:
    with _active_realtime_sessions_lock:
        _active_realtime_sessions.pop(session_id, None)


def _sse_pack(event_type: str, payload: dict) -> str:
    # SSE format: event: <name>\n data: <json>\n\n
    import json
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

from backend.services.text2audio_service import text_to_audio
from backend.models.db_models import PromptHistory
from backend.extensions import db

audio_bp = Blueprint("audio_bp", __name__)

ALLOWED_AUDIO = {"mp3", "wav", "m4a", "ogg", "flac", "webm"}


def allowed_audio(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO


# ─── Audio to Text ────────────────────────────────────────────────────────────

@audio_bp.route("/audio2text")
def audio2text_page():
    return render_template("audio2text.html")


@audio_bp.route("/api/audio-to-text", methods=["POST"])
def audio_to_text_route():
    try:
        if "audio" not in request.files:
            return jsonify({"success": False, "error": "No audio file uploaded"}), 400

        file = request.files["audio"]

        if not allowed_audio(file.filename):
            return jsonify({"success": False, "error": "Invalid audio format. Use MP3, WAV, M4A, OGG"}), 400

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        transcript = audio_to_text(upload_path)

        record = PromptHistory(operation_type="audio-to-text", prompt=f"[audio] {file.filename}", output=transcript)
        db.session.add(record)
        db.session.commit()

        return jsonify({"success": True, "transcript": transcript})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Text to Audio ────────────────────────────────────────────────────────────

@audio_bp.route("/text2audio")
def text2audio_page():
    return render_template("text2audio.html")


@audio_bp.route("/api/text-to-audio", methods=["POST"])
def text_to_audio_route():
    try:
        data = request.get_json()
        if not data or not data.get("text"):
            return jsonify({"success": False, "error": "No text provided"}), 400

        text = data["text"].strip()
        lang = data.get("lang", "en")

        filename = f"tts_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(current_app.config["GENERATED_FOLDER"], filename)

        text_to_audio(text, output_path, lang)

        record = PromptHistory(operation_type="text-to-audio", prompt=text[:200], output_file=filename)
        db.session.add(record)
        db.session.commit()

        return jsonify({"success": True, "audio_url": f"/api/generated/{filename}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── History route ────────────────────────────────────────────────────────────

@audio_bp.route("/api/history")
def get_history():
    records = PromptHistory.query.order_by(PromptHistory.created_at.desc()).limit(50).all()
    return jsonify({"success": True, "history": [r.to_dict() for r in records]})


#    LIVE Audio  Text (SSE)


@audio_bp.route("/api/audio-to-text/live-stream")
def audio_to_text_live_stream():
    """SSE endpoint for live microphone transcription.

    Client provides `session_id` query param.
    Streaming events:
      - partial: {text}
      - final: {text}
      - error: {error}

    Graceful client disconnect handling: generator finalization stops the session.
    """

    import time
    from flask import Response, request

    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"success": False, "error": "Missing session_id"}), 400

    rt = _create_session(session_id)

    def generate():
        import time
        import logging
        from flask import request as flask_request

        log = logging.getLogger(__name__)
        try:
            log.info("[sse] connected session_id=%s ip=%s", session_id, flask_request.remote_addr)

            # Emit an early status FIRST so EventSource sees bytes before heavy work.
            yield "event: status\ndata: {\"message\":\"connected\"}\n\n"
            yield ": ping\n\n"

            log.info("[sse] starting rt.start() session_id=%s", session_id)

            # Start transcription session
            rt.start()

            # Notify listening
            yield _sse_pack("status", {"message": "listening"})

            last_yield = time.time()


            # Keep heartbeats flowing even when no transcript events arrive.
            hb_interval_s = 2.5

            for evt in rt.transcript_stream():
                # If no transcript arrives for a while, push heartbeat.
                now = time.time()
                if now - last_yield >= hb_interval_s:
                    log.debug("[sse] heartbeat session_id=%s", session_id)
                    yield ": ping\n\n"
                    last_yield = now

                etype = (evt.get("type") or "").strip()
                if etype == "partial":
                    log.info("[sse] partial session_id=%s text_len=%s", session_id, len(evt.get("text", "")))
                    yield _sse_pack("partial", {"text": evt.get("text", "")})
                    last_yield = time.time()
                elif etype == "final":
                    log.info("[sse] final session_id=%s", session_id)
                    yield _sse_pack("final", {"text": evt.get("text", "")})
                    last_yield = time.time()
                    break
                elif etype == "error":
                    log.error("[sse] error session_id=%s err=%s", session_id, evt.get("error"))
                    yield _sse_pack("error", {"error": evt.get("error", "Unknown error")})
                    last_yield = time.time()
                    break
                else:
                    # keep unknown event types from breaking stream
                    log.warning("[sse] unknown event_type=%s session_id=%s", etype, session_id)
                    yield _sse_pack("status", {"message": evt})
                    last_yield = time.time()

        except GeneratorExit:
            log.info("[sse] client disconnected session_id=%s", session_id)
        except Exception as e:
            log.exception("[sse] stream exception session_id=%s", session_id)
            try:
                yield _sse_pack("error", {"error": str(e)})
            except Exception:
                pass
        finally:
            log.info("[sse] stream closed session_id=%s", session_id)
            # Ensure microphone is released
            try:
                rt.stop()
            except Exception:
                pass
            _remove_session(session_id)


    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        direct_passthrough=True,
    )




@audio_bp.route("/api/audio-to-text/live-stop", methods=["POST"])
def audio_to_text_live_stop():
    """Stops an active live transcription session."""

    from flask import request, jsonify

    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or request.args.get("session_id") or "").strip()

    if not session_id:
        return jsonify({"success": False, "error": "Missing session_id"}), 400

    rt = _get_session(session_id)
    if rt is None:
        return jsonify({"success": True})

    try:
        rt.stop()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        _remove_session(session_id)

    return jsonify({"success": True})

