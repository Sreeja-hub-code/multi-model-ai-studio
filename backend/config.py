import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///multimodal.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Kept for backwards compatibility with older code paths.
    # Text generation is now fully local in `backend/services/text_service.py`.
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    HF_TOKEN = os.getenv("HF_TOKEN", "")

    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    GENERATED_FOLDER = os.path.join(os.path.dirname(__file__), "generated")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
    ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac"}
