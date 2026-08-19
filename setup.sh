#!/usr/bin/env bash
# ============================================================
# setup.sh (v4) — гибридный конвейер: Gemini API (сценарий, hero-сцены) + локальный FLUX (рядовые сцены)
#
# Архитектура:
# - Сценарий + Hero-сцены: Gemini API (требует ключ).
# - Рядовые сцены: FLUX.1-schnell (локально на GPU).
# - Анимация/Звук: Wan 2.2, XTTS, whisper (локально на GPU).
# ============================================================
set -euo pipefail

PROJECT_ROOT="$HOME/aipipe"
VENV_DIR="$PROJECT_ROOT/venv"
MODELS_DIR="$PROJECT_ROOT/models"
REPOS_DIR="$PROJECT_ROOT/repos"
LOG() { echo -e "\n\033[1;32m[SETUP]\033[0m $1"; }
FAIL() { echo -e "\n\033[1;31m[ОШИБКА]\033[0m $1"; exit 1; }

mkdir -p "$PROJECT_ROOT" "$MODELS_DIR" "$REPOS_DIR"
cd "$PROJECT_ROOT"

# ------------------------------------------------------------
# 0. config.env (возвращаем модели Gemini Image для hero-сцен)
# ------------------------------------------------------------
if [ ! -f "$PROJECT_ROOT/config.env" ]; then
  cat > "$PROJECT_ROOT/config.env" << 'EOF'
# === ВПИШИ СВОИ ЗНАЧЕНИЯ ПЕРЕД ЗАПУСКОМ PIPELINE ===

# Gemini API — используется для сценариев и hero-сцен
export GEMINI_API_KEY="ВСТАВЬ_КЛЮЧ_ИЗ_aistudio.google.com/apikey"

# Голос XTTS
export VOICE_SAMPLE="$HOME/aipipe/assets/voice_sample.wav"

# Модели Gemini
export GEMINI_TEXT_MODEL="gemini-3-pro-preview"
export GEMINI_IMAGE_MODEL_FAST="gemini-2.5-flash-image"      # для --quality fast
export GEMINI_IMAGE_MODEL_MAX="gemini-3-pro-image-preview"   # для --quality max
EOF
  LOG "Создан config.env — ОБЯЗАТЕЛЬНО впиши GEMINI_API_KEY."
fi

# ... (создание папок остается прежним) ...

# ------------------------------------------------------------
# 1. Системные зависимости (без изменений)
# ------------------------------------------------------------
LOG "Ставлю системные пакеты"
sudo apt-get update -y
sudo apt-get install -y \
  python3.11 python3.11-venv python3-pip git git-lfs wget curl \
  ffmpeg libsndfile1 build-essential ninja-build
git lfs install
nvidia-smi || FAIL "nvidia-smi не работает — проверь драйвер GPU на VM."

# ------------------------------------------------------------
# 2. Python venv и PyTorch (без изменений)
# ------------------------------------------------------------
LOG "Создаю venv и ставлю PyTorch"
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# 3. Python-зависимости (Gemini + Diffusers для FLUX)
# ------------------------------------------------------------
LOG "Ставлю SDK (google-genai, diffusers)"
pip install google-genai pillow tenacity python-dotenv
pip install diffusers transformers accelerate safetensors

# ------------------------------------------------------------
# 4. Локальная модель FLUX для рядовых сцен
# ------------------------------------------------------------
LOG "Скачиваю модель FLUX.1-schnell для быстрой локальной генерации..."
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir "$MODELS_DIR/flux1-schnell" --local-dir-use-symlinks False

# ... (секции Wan 2.2, XTTS, faster-whisper остаются без изменений) ...

# ------------------------------------------------------------
# 5. Wan 2.2 (анимация)
# ------------------------------------------------------------
LOG "Клонирую и ставлю Wan 2.2"
cd "$REPOS_DIR"
[ -d Wan2.2 ] || git clone https://github.com/Wan-Video/Wan2.2.git
pip install -r Wan2.2/requirements.txt || true
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir "$MODELS_DIR/wan2.2-i2v"

# ------------------------------------------------------------
# 6. XTTS-v2 (озвучка)
# ------------------------------------------------------------
LOG "Ставлю coqui TTS (XTTS-v2)"
pip install TTS
python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# ------------------------------------------------------------
# 7. faster-whisper (субтитры)
# ------------------------------------------------------------
LOG "Ставлю faster-whisper"
pip install faster-whisper

# ------------------------------------------------------------
# 8. Freeze зависимостей
# ------------------------------------------------------------
LOG "Пишу freeze зависимостей"
pip freeze > "$PROJECT_ROOT/requirements.lock.txt"
nvidia-smi > "$PROJECT_ROOT/gpu_info.txt"

LOG "ГОТОВО. Дальше:
1) Впиши GEMINI_API_KEY в config.env.
2) Положи assets/voice_sample.wav.
3) Запусти генерацию: python3 pipeline.py --topic '...' --series '...' --character '...' --new-character '...'
"