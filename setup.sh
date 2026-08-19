#!/usr/bin/env bash
# ============================================================
# setup.sh — установка полностью автономного локального AI-стека под
# рисованные ролики (единый стиль + единый персонаж). Без внешних API/квот:
# FLUX.1-dev + style-LoRA + PuLID/IP-Adapter для картинок и консистентности,
# Wan 2.2 для анимации hero-сцен, Qwen (Ollama) для сценариев.
#
# Запуск:  bash setup.sh 2>&1 | tee setup.log
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
# 0. config.env
# ------------------------------------------------------------
if [ ! -f "$PROJECT_ROOT/config.env" ]; then
  cat > "$PROJECT_ROOT/config.env" << 'EOF'
# === ВПИШИ СВОИ ЗНАЧЕНИЯ ПЕРЕД ЗАПУСКОМ PIPELINE ===
# Полностью автономный локальный стек — никаких внешних API/квот.

# Hugging Face токен — нужен для скачки FLUX.1-dev (gated модель,
# прими условия лицензии на странице модели на HF перед скачкой)
export HF_TOKEN="hf_ВСТАВЬ_СВОЙ_ТОКЕН"

export CIVITAI_API_KEY="ВСТАВЬ_ЕСЛИ_НУЖНО"        # civitai.com/user/account, для скачки LoRA скриптом
export STYLE_LORA_URL=""                          # прямая ссылка на .safetensors с Civitai
export VOICE_SAMPLE="$HOME/aipipe/assets/voice_sample.wav"   # 10-20 сек чистой речи для XTTS

export QWEN_MODEL="qwen2.5:32b"    # если не влезет в VRAM вместе с FLUX/Wan — возьми qwen2.5:14b
EOF
  LOG "Создан config.env — ОБЯЗАТЕЛЬНО впиши HF_TOKEN перед запуском pipeline.py"
fi
source "$PROJECT_ROOT/config.env"
mkdir -p "$PROJECT_ROOT/assets" "$PROJECT_ROOT/output" "$PROJECT_ROOT/scenes" \
         "$PROJECT_ROOT/characters" "$PROJECT_ROOT/state"

# ------------------------------------------------------------
# 1. Системные зависимости
# ------------------------------------------------------------
LOG "Ставлю системные пакеты"
sudo apt-get update -y
sudo apt-get install -y \
  python3.11 python3.11-venv python3-pip git git-lfs wget curl \
  ffmpeg libsndfile1 build-essential ninja-build

nvidia-smi || FAIL "nvidia-smi не работает — проверь драйвер GPU на VM перед продолжением"

# ------------------------------------------------------------
# 2. Python venv
# ------------------------------------------------------------
LOG "Создаю venv"
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# ------------------------------------------------------------
# 3. PyTorch (под FLUX/Wan/XTTS/whisper)
# ------------------------------------------------------------
LOG "Ставлю PyTorch с CUDA"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA недоступна'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# ------------------------------------------------------------
# 4. Diffusers-стек: FLUX.1-dev + style-LoRA + PuLID/IP-Adapter
# ------------------------------------------------------------
LOG "Ставлю diffusers, transformers, accelerate, peft, xformers"
pip install diffusers transformers accelerate peft sentencepiece protobuf \
  huggingface_hub safetensors einops xformers tenacity pillow

huggingface-cli login --token "$HF_TOKEN" || FAIL "HF login не прошёл — проверь HF_TOKEN в config.env"

LOG "Скачиваю веса FLUX.1-dev (gated модель, нужен принятый лицензионный доступ на HF)"
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir "$MODELS_DIR/flux1-dev"

# ------------------------------------------------------------
# 5. Style-LoRA (Civitai)
# ------------------------------------------------------------
LOG "Скачиваю style-LoRA (если STYLE_LORA_URL задан в config.env)"
mkdir -p "$MODELS_DIR/loras"
if [ -n "${STYLE_LORA_URL:-}" ]; then
  wget --header="Authorization: Bearer $CIVITAI_API_KEY" -O "$MODELS_DIR/loras/style_lora.safetensors" "$STYLE_LORA_URL"
else
  echo "⚠ STYLE_LORA_URL пуст — впиши ссылку в config.env и перезапусти этот блок вручную"
fi

# ------------------------------------------------------------
# 6. IP-Adapter / PuLID для консистентности персонажа
# ------------------------------------------------------------
LOG "Клонирую FLUX IP-Adapter (XLabs-AI) и PuLID-FLUX"
cd "$REPOS_DIR"
[ -d x-flux ] || git clone https://github.com/XLabs-AI/x-flux.git
[ -d PuLID ] || git clone https://github.com/ToTheBeginning/PuLID.git
pip install -r x-flux/requirements.txt || true
pip install -r PuLID/requirements.txt || true

LOG "Скачиваю веса IP-Adapter/PuLID для FLUX"
huggingface-cli download XLabs-AI/flux-ip-adapter --local-dir "$MODELS_DIR/flux-ip-adapter"
huggingface-cli download guozinan/PuLID --local-dir "$MODELS_DIR/pulid" || echo "⚠ проверь актуальный repo_id PuLID-FLUX на HF на момент установки"

# ------------------------------------------------------------
# 7. Wan 2.2 (точечная анимация hero-сцен)
# ------------------------------------------------------------
LOG "Клонирую и ставлю Wan 2.2"
cd "$REPOS_DIR"
[ -d Wan2.2 ] || git clone https://github.com/Wan-Video/Wan2.2.git
pip install -r Wan2.2/requirements.txt || true
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir "$MODELS_DIR/wan2.2-i2v" || \
  echo "⚠ проверь актуальный repo_id Wan2.2 на HF на момент установки — мог измениться"

# ------------------------------------------------------------
# 8. XTTS-v2 (озвучка)
# ------------------------------------------------------------
LOG "Ставлю coqui TTS (XTTS-v2)"
pip install TTS
python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')" || \
  echo "⚠ первая инициализация XTTS скачает веса — если упало, перезапусти строку вручную"

# ------------------------------------------------------------
# 9. faster-whisper (субтитры)
# ------------------------------------------------------------
LOG "Ставлю faster-whisper"
pip install faster-whisper

# ------------------------------------------------------------
# 10. Локальный Qwen (Ollama) — дефолтный backend для сценариев,
#    чтобы этот шаг не зависел от интернета/API-ключа
# ------------------------------------------------------------
LOG "Ставлю Ollama + Qwen2.5 (локальный сценарист, TEXT_BACKEND=qwen_local по умолчанию)"
curl -fsSL https://ollama.com/install.sh | sh
ollama pull "${QWEN_MODEL:-qwen2.5:32b}" || {
  echo "⚠ ${QWEN_MODEL} не встал (не хватило VRAM?) — пробую qwen2.5:14b"
  ollama pull qwen2.5:14b
  echo "  -> если взяло 14b, поставь QWEN_MODEL=qwen2.5:14b в config.env"
}

LOG "Проверка: Qwen отвечает?"
ollama run "${QWEN_MODEL:-qwen2.5:32b}" "Ответь одним словом: работаешь?" || \
  echo "⚠ Qwen не ответил — проверь 'ollama list' и 'ollama serve' вручную"

# ------------------------------------------------------------
# 11. Freeze зависимостей
# ------------------------------------------------------------
LOG "Пишу freeze зависимостей"
pip freeze > "$PROJECT_ROOT/requirements.lock.txt"
nvidia-smi > "$PROJECT_ROOT/gpu_info.txt"

LOG "ГОТОВО. Дальше:
1) Впиши STYLE_LORA_URL в config.env (выбери LoRA на Civitai под нужный стиль)
2) Положи assets/voice_sample.wav (10-20 сек чистой речи)
3) Заполни characters/registry.json (см. README) под своих персонажей
4) Тест: python3 pipeline.py --topic 'миф про Икара' --series mythology --duration 90 --quality fast
5) Сделай снапшот диска ДО первого реального прогона (см. README.md)"
