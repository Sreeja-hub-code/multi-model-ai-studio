import os
import uuid
from flask import Blueprint, request, jsonify, render_template, send_file, current_app
from werkzeug.utils import secure_filename
from backend.services.text2image_service import generate_image
from backend.services.image2text_service import image_to_text
from backend.services.image2image_ultrafast_service import image_to_image
from backend.models.db_models import PromptHistory
from backend.extensions import db

image_bp = Blueprint("image_bp", __name__)

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "webp", "gif"}


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE


# ─── Text to Image ──────────────────────────────────────────────────────────

@image_bp.route("/text2image")
def text2image_page():
    return render_template("text2image.html")


@image_bp.route("/api/text-to-image", methods=["POST"])
def text_to_image():
    try:
        data = request.get_json(silent=True) or {}
        if not data.get("prompt"):
            return jsonify({"success": False, "ok": False, "error": "No prompt provided"}), 400

        prompt = data["prompt"].strip()
        filename = f"{uuid.uuid4().hex}.png"
        output_path = os.path.join(current_app.config["GENERATED_FOLDER"], filename)

        generated_path = generate_image(prompt, output_path)

        record = PromptHistory(operation_type="text-to-image", prompt=prompt, output_file=filename)
        db.session.add(record)
        db.session.commit()

        return jsonify({"success": True, "ok": True, "image_url": f"/api/generated/{filename}"})

    except Exception as e:
        # Always return JSON, never crash the server
        return jsonify({"success": False, "ok": False, "error": str(e)}), 500



# ─── Image to Text ───────────────────────────────────────────────────────────

@image_bp.route("/image2text")
def image2text_page():
    return render_template("image2text.html")


@image_bp.route("/api/image-to-text", methods=["POST"])
def image_to_text_route():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["image"]
        mode = request.form.get("mode", "describe")

        if not allowed_image(file.filename):
            return jsonify({"success": False, "error": "Invalid image format"}), 400

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        result = image_to_text(upload_path, mode)

        record = PromptHistory(operation_type="image-to-text", prompt=f"[{mode}] {file.filename}", output=result)
        db.session.add(record)
        db.session.commit()

        return jsonify({"success": True, "response": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Image to Image ───────────────────────────────────────────────────────────

@image_bp.route("/img2img")
def img2img_page():
    return render_template("img2img.html")


@image_bp.route("/api/image-to-image", methods=["POST"])
def image_to_image_route():
    try:
        print("[ROUTE] image-to-image route called")
        print("[ROUTE] using service:", image_to_image.__module__)

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["image"]
        prompt = request.form.get("prompt", "").strip()

        if not prompt:
            return jsonify({"success": False, "error": "No transformation prompt provided"}), 400
        if not allowed_image(file.filename):
            return jsonify({"success": False, "error": "Invalid image format"}), 400

        in_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        print("[LOCAL IMG2IMG] local pipeline executing")
        upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], in_filename)
        file.save(upload_path)

        out_filename = f"i2i_{uuid.uuid4().hex}.png"
        output_path = os.path.join(current_app.config["GENERATED_FOLDER"], out_filename)

        image_to_image(upload_path, prompt, output_path)

        record = PromptHistory(operation_type="image-to-image", prompt=prompt, output_file=out_filename)
        db.session.add(record)
        db.session.commit()

        return jsonify({"success": True, "image_url": f"/api/generated/{out_filename}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Serve generated files ─────────────────────────────────────────────────

@image_bp.route("/api/generated/<filename>")
def serve_generated(filename):
    path = os.path.join(current_app.config["GENERATED_FOLDER"], filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path)
