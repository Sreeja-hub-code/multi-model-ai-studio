# 🤖 Multimodal AI Studio

A complete Flask web app with 6 AI modes powered by Claude, Whisper, Stable Diffusion, and gTTS.

## ✅ Features

| Mode | Model Used | Requires |
|---|---|---|
| 💬 Text → Text | Claude claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| 🎨 Text → Image | Stable Diffusion XL (HuggingFace) | `HF_TOKEN` |
| 🔍 Image → Text | Claude Vision | `ANTHROPIC_API_KEY` |
| 🎙️ Audio → Text | OpenAI Whisper (local) | `pip install openai-whisper` |
| 🔊 Text → Audio | gTTS (Google TTS) | `pip install gtts` |
| 🖼️ Image → Image | instruct-pix2pix (HuggingFace) | `HF_TOKEN` |

---

## 🚀 Quick Start

### 1. Clone / extract the project

```bash
cd multimodal-ai
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Set your API keys

Edit the `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...your key here...
HF_TOKEN=hf_...your token here...
```

**Get keys:**
- Anthropic: https://console.anthropic.com
- HuggingFace: https://huggingface.co/settings/tokens

### 5. Run the app

```bash
python run.py
```

Open http://localhost:5000 in your browser.

---

## 📁 Project Structure

```
multimodal-ai/
├── run.py                          ← Start here
├── .env                            ← Your API keys
├── backend/
│   ├── app.py                      ← Flask app factory
│   ├── config.py                   ← Configuration
│   ├── extensions.py               ← SQLAlchemy
│   ├── requirements.txt
│   ├── models/
│   │   └── db_models.py            ← Database models
│   ├── routes/
│   │   ├── text_routes.py          ← Text API
│   │   ├── image_routes.py         ← Image API
│   │   └── audio_routes.py         ← Audio API
│   ├── services/
│   │   ├── text_service.py         ← Claude text
│   │   ├── text2image_service.py   ← Stable Diffusion
│   │   ├── image2text_service.py   ← Claude vision
│   │   ├── audio2text_service.py   ← Whisper
│   │   ├── text2audio_service.py   ← gTTS
│   │   └── image2image_service.py  ← pix2pix
│   ├── uploads/                    ← User uploads
│   └── generated/                  ← AI outputs
└── frontend/
    ├── templates/
    │   ├── base.html
    │   ├── index.html
    │   ├── text2text.html
    │   ├── text2image.html
    │   ├── image2text.html
    │   ├── audio2text.html
    │   ├── text2audio.html
    │   └── img2img.html
    └── static/
        ├── css/style.css
        └── js/app.js
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/text-to-text` | `{"prompt": "..."}` |
| POST | `/api/text-to-image` | `{"prompt": "..."}` |
| POST | `/api/image-to-text` | multipart: `image`, `mode` |
| POST | `/api/audio-to-text` | multipart: `audio` |
| POST | `/api/text-to-audio` | `{"text": "...", "lang": "en"}` |
| POST | `/api/image-to-image` | multipart: `image`, `prompt` |
| GET | `/api/history` | All past generations |
| GET | `/api/health` | API key status |

---

## 🌐 Deployment

### Render / Railway
1. Push to GitHub
2. Set environment variables (`ANTHROPIC_API_KEY`, `HF_TOKEN`)
3. Set start command: `python run.py`

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r backend/requirements.txt
EXPOSE 5000
CMD ["python", "run.py"]
```

---

## ⚠️ Troubleshooting

**"Failed to fetch"** → You're accessing the API from a browser widget that blocks cross-origin requests. Always run this as a local Flask app (`python run.py`) and open http://localhost:5000.

**Image generation slow** → HuggingFace free tier cold-starts can take 60–90 seconds. First request may be slow; subsequent ones are faster.

**Whisper not working** → Run: `pip install openai-whisper` then restart.

**gTTS error** → Requires internet connection (calls Google TTS servers).
