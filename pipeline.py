#!/usr/bin/env python3
"""
pipeline.py (v4.1) — гибридный конвейер, настроенный под стиль "motion comic".

Архитектура:
- Сценарий + Hero-сцены: Gemini API.
- Рядовые сцены: Локальный FLUX.1-schnell.
- Анимация: Замедленный zoom/pan и микро-анимации через Wan 2.2.
"""

import os
import io
import json
import time
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from google import genai
from google.genai import types as gtypes
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
from diffusers import FluxPipeline
from PIL import Image

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
SCENES = ROOT / "scenes"
# ... (остальные пути)

# 1. ОБНОВЛЕНИЕ STYLE_SUFFIX (в соответствии с вашими инструкциями)
STYLE_SUFFIX = (
    "in the style of 90s dark fantasy anime, highly detailed digital painting, "
    "2D illustration, graphic novel aesthetic, muted earthy color palette, "
    "cinematic lighting, melancholic atmosphere, highly detailed expressive eyes "
    "--no 3d, photorealism, hyperrealism, bright neon colors"
)

# ... (Класс LocalFluxGenerator и датаклассы остаются без изменений) ...

# 2. НАСТРОЙКА АНИМАЦИИ (в соответствии с вашими инструкциями)
def animate_zoompan(scene: Scene, seconds: float = 6.0) -> Path:
    out_path = SCENES / f"scene_{scene.index:03d}_anim.mp4"
    # Уменьшаем скорость зума для более медленного, медитативного эффекта
    zoom_expr = "min(zoom+0.0010,1.10)"
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", scene.image_path,
        "-vf", f"zoompan=z='{zoom_expr}':d={int(seconds*30)}:s=576x1280:fps=30", # Устанавливаем FPS и разрешение
        "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "3000k", # Устанавливаем битрейт
        str(out_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path

def animate_wan(scene: Scene, seconds: float = 5.0) -> Path:
    out_path = SCENES / f"scene_{scene.index:03d}_anim.mp4"
    # Добавляем промпты для микро-анимаций
    motion_prompt = f"{scene.visual_prompt}, subtle movement, hair softly blowing in the wind, cinemagraph"
    cmd = [
        "python3", str(ROOT / "repos" / "Wan2.2" / "generate.py"),
        "--task", "i2v-A14B",
        "--ckpt_dir", str(MODELS_DIR / "wan2.2-i2v"),
        "--image", scene.image_path,
        "--prompt", motion_prompt, # Используем новый промпт
        "--frame_num", str(int(seconds * 16)),
        "--save_file", str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path

# ... (все остальные функции остаются без изменений, как в v4) ...

def main():
    # ... (логика main остается без изменений) ...
    print("Запуск конвейера v4.1 (motion comic style)")
    # ...

if __name__ == "__main__":
    main()
