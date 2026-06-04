import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify
from flask_cors import CORS

from backend.config import Config
from backend.extensions import db
from backend.routes.text_routes import text_bp
from backend.routes.image_routes import image_bp
from backend.routes.audio_routes import audio_bp


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "../frontend/templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "../frontend/static"),
    )

    app.config.from_object(Config)

    # Create folders if they don't exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["GENERATED_FOLDER"], exist_ok=True)

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    db.init_app(app)

    app.register_blueprint(text_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(audio_bp)

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        api_key = app.config.get("ANTHROPIC_API_KEY", "")
        hf_token = app.config.get("HF_TOKEN", "")
        return jsonify({
            "status": "ok",
            "anthropic_key": "✅ Set" if api_key and api_key != "your_anthropic_api_key_here" else "❌ Missing",
            "hf_token": "✅ Set" if hf_token and hf_token != "your_huggingface_token_here" else "❌ Missing (image gen will fail)",
        })

    with app.app_context():
        db.create_all()

    return app


app = create_app()

if __name__ == "__main__":
    # SSE long-lived connections are sensitive to reloader interference.
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)

