from flask import Blueprint, request, jsonify, render_template
from backend.services.text_service import generate_text
from backend.models.db_models import PromptHistory
from backend.extensions import db

text_bp = Blueprint("text_bp", __name__)


@text_bp.route("/text2text")
def text_page():
    return render_template("text2text.html")


@text_bp.route("/api/text-to-text", methods=["POST"])
def text_to_text():
    try:
        data = request.get_json()
        if not data or not data.get("prompt"):
            return jsonify({"success": False, "error": "No prompt provided"}), 400

        prompt = data["prompt"].strip()
        system = data.get("system", "You are a helpful, accurate AI assistant. Give clear, concise responses.")

        output = generate_text(prompt, system=system)

        # Save to DB
        record = PromptHistory(
            operation_type="text-to-text",
            prompt=prompt,
            output=output,
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({"ok": True, "success": True, "response": output})

    except Exception as e:
        return jsonify({"ok": False, "success": False, "error": str(e)}), 500

