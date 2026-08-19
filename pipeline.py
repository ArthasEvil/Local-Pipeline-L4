#!/usr/bin/env python3
"""
pipeline.py — оркестратор генерации ролика. Полностью автономно, без
внешних API/квот.

Qwen (Ollama, сценарий по сценам)
  -> FLUX.1-dev + style-LoRA + PuLID/IP-Adapter (референс персонажа) -> кадры сцен
  -> ffmpeg zoompan (дёшево) ИЛИ Wan 2.2 I2V (дорого, только hero-сцены)
  -> XTTS-v2 (голос, сохранённый speaker_wav)
  -> faster-whisper (word-level таймкоды)
  -> ffmpeg ASS burn + финальная сборка + серийный intro/outro

Ключевые отличия от v1:
- Никакого локального FLUX/LoRA/PuLID/IP-Adapter — консистентность персонажа
  и стиль закрывает FLUX + style-LoRA + PuLID/IP-Adapter через reference-image conditioning.
- Реестр персонажей (characters/registry.json) — один раз генеришь reference
  sheet персонажа, переиспользуешь во всех роликах серии.
- Флаг --quality fast|max переключает число шагов/guidance генерации FLUX
  (больше шагов + upscale на max) и долю hero-сцен на Wan, не трогая код.
- Resume: прогресс сцены сохраняется в state/<run_id>.json, повторный запуск
  с тем же --run-id продолжает с места обрыва (VM отвалилась — не начинаем
  ролик с нуля).
- Серии (--series) — у каждой категории свой intro/outro темплейт и цветовая
  плашка, чтобы зритель узнавал линейку с первого кадра.

Запуск:
  python3 pipeline.py --topic "Икар и крылья из воска" --series mythology \
      --duration 90 --quality fast --character icarus_default
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

import torch
from diffusers import FluxPipeline
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image

# ============================================================
# Пути и конфиг
# ============================================================
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
SCENES = ROOT / "scenes"
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"
REPOS = ROOT / "repos"
CHARACTERS_DIR = ROOT / "characters"
STATE_DIR = ROOT / "state"
SERIES_DIR = ROOT / "series_templates"

for d in (ASSETS, SCENES, OUTPUT, CHARACTERS_DIR, STATE_DIR, SERIES_DIR):
    d.mkdir(exist_ok=True)

VOICE_SAMPLE = Path(os.environ.get("VOICE_SAMPLE", ASSETS / "voice_sample.wav"))

# Текст (сценарий) — локальный Qwen через Ollama. Полностью автономно,
# без внешних API и квот.
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen2.5:32b")

# Картинки — локальный FLUX.1-dev + style-LoRA + PuLID/IP-Adapter для
# консистентности персонажа. Полностью автономно, без внешних API/квот.
STYLE_LORA = MODELS / "loras" / "style_lora.safetensors"
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", "28"))
IMAGE_GUIDANCE = float(os.environ.get("IMAGE_GUIDANCE", "3.5"))

STYLE_SUFFIX = (
    ", painterly storybook illustration, warm muted color palette, "
    "soft cinematic lighting, detailed linework, semi-realistic "
    "anime-influenced style, consistent art direction"
)

_flux_pipe = None
_character_pipe = None
_use_pulid = False


def get_flux_pipeline():
    """Ленивая загрузка FLUX — грузим один раз на весь прогон, не на каждую сцену."""
    global _flux_pipe, _character_pipe, _use_pulid
    if _flux_pipe is not None:
        return _flux_pipe, _character_pipe, _use_pulid

    pipe = FluxPipeline.from_pretrained(
        str(MODELS / "flux1-dev"), torch_dtype=torch.bfloat16,
    ).to("cuda")

    if STYLE_LORA.exists():
        pipe.load_lora_weights(str(STYLE_LORA))
    else:
        print(f"⚠ style-LoRA не найдена по пути {STYLE_LORA} — генерация пойдёт без неё")

    try:
        import sys
        sys.path.insert(0, str(REPOS / "PuLID"))
        from pulid_flux import PuLIDPipeline  # имя модуля сверить в актуальном репо PuLID
        character_pipe = PuLIDPipeline(base_pipe=pipe, weights_dir=str(MODELS / "pulid"))
        _flux_pipe, _character_pipe, _use_pulid = pipe, character_pipe, True
    except Exception as e:
        print(f"⚠ PuLID не поднялся ({e}) — fallback на IP-Adapter")
        pipe.load_ip_adapter(str(MODELS / "flux-ip-adapter"))
        _flux_pipe, _character_pipe, _use_pulid = pipe, None, False

    return _flux_pipe, _character_pipe, _use_pulid


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=3, max=30))
def _generate_image(prompt: str, ref_image_path: Optional[Path] = None) -> Image.Image:
    pipe, character_pipe, use_pulid = get_flux_pipeline()

    if ref_image_path and use_pulid:
        return character_pipe.generate(
            prompt=prompt, id_image=str(ref_image_path),
            num_inference_steps=IMAGE_STEPS, guidance_scale=IMAGE_GUIDANCE,
        )
    elif ref_image_path:
        return pipe(
            prompt=prompt, ip_adapter_image=str(ref_image_path),
            num_inference_steps=IMAGE_STEPS, guidance_scale=IMAGE_GUIDANCE,
        ).images[0]
    else:
        return pipe(
            prompt=prompt, num_inference_steps=IMAGE_STEPS, guidance_scale=IMAGE_GUIDANCE,
        ).images[0]


# ============================================================
# Структуры данных
# ============================================================
@dataclass
class Scene:
    index: int
    narration: str
    visual_prompt: str
    is_hero: bool = False
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    status: str = "pending"  # pending -> image_done -> anim_done


@dataclass
class Spec:
    run_id: str
    topic: str
    series: str
    duration_sec: int
    character_id: Optional[str]
    quality: str
    scenes: list = field(default_factory=list)

    def to_json(self):
        return json.dumps({
            **{k: v for k, v in asdict(self).items() if k != "scenes"},
            "scenes": [asdict(s) for s in self.scenes]
        }, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(data: dict) -> "Spec":
        scenes = [Scene(**s) for s in data.pop("scenes")]
        spec = Spec(**data, scenes=[])
        spec.scenes = scenes
        return spec


# ============================================================
# State / resume
# ============================================================
def state_path(run_id: str) -> Path:
    return STATE_DIR / f"{run_id}.json"


def save_state(spec: Spec):
    state_path(spec.run_id).write_text(spec.to_json(), encoding="utf-8")


def load_state(run_id: str) -> Optional[Spec]:
    p = state_path(run_id)
    if not p.exists():
        return None
    return Spec.from_json(json.loads(p.read_text(encoding="utf-8")))


# ============================================================
# Реестр персонажей — один раз генерим reference sheet, переиспользуем
# ============================================================
def character_registry_path() -> Path:
    return CHARACTERS_DIR / "registry.json"


def load_character_registry() -> dict:
    p = character_registry_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_character_registry(reg: dict):
    character_registry_path().write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def create_character(character_id: str, description: str) -> Path:
    """
    Генерирует reference sheet персонажа (фронт + 45° + профиль в одном кадре)
    один раз локально через FLUX, сохраняет в characters/<id>.png, регистрирует
    в registry.json. Это единственный кадр, к которому будут "привязываться"
    все остальные через PuLID/IP-Adapter — стоит генерить с чуть большим числом
    шагов, чем обычные сцены.
    """
    reg = load_character_registry()
    if character_id in reg:
        print(f"[character] '{character_id}' уже в реестре -> {reg[character_id]['path']}")
        return Path(reg[character_id]["path"])

    prompt = (
        f"Character reference sheet, three views in one image: front view, "
        f"3/4 view, side profile. Character description: {description}."
        f"{STYLE_SUFFIX}. Neutral pose, neutral expression, plain background, "
        f"consistent lighting across all three views."
    )
    image = _generate_image(prompt)

    out_path = CHARACTERS_DIR / f"{character_id}.png"
    image.save(out_path)

    reg[character_id] = {"path": str(out_path), "description": description}
    save_character_registry(reg)
    print(f"[character] '{character_id}' создан -> {out_path}")
    return out_path


# ============================================================
# Серийные intro/outro темплейты
# ============================================================
DEFAULT_SERIES_TEMPLATES = {
    "mythology": {
        "intro_prompt": "Ornate title card, ancient Greek marble motif, gold "
                         "engraved text placeholder, dramatic clouds background",
        "accent_color": "#C9A227",
    },
    "history": {
        "intro_prompt": "Weathered parchment title card, sepia tones, wax seal "
                         "motif, candlelight vignette",
        "accent_color": "#8B5E34",
    },
    "spiritual": {
        "intro_prompt": "Minimalist title card, single glowing eye motif, deep "
                         "indigo background, soft golden particles",
        "accent_color": "#4B3F72",
    },
    "original": {
        "intro_prompt": "Modern painterly title card, warm gradient background, "
                         "subtle brush texture",
        "accent_color": "#A63A3A",
    },
}


def get_series_template(series: str) -> dict:
    templates_file = SERIES_DIR / "templates.json"
    if templates_file.exists():
        custom = json.loads(templates_file.read_text(encoding="utf-8"))
        if series in custom:
            return custom[series]
    return DEFAULT_SERIES_TEMPLATES.get(series, DEFAULT_SERIES_TEMPLATES["original"])


# ============================================================
# Шаг 1. Генерация спек-документа — локальный Qwen (Ollama)
# ============================================================
def _spec_prompt(series: str, topic: str, n_scenes: int, duration_sec: int) -> str:
    return f"""Ты — сценарист коротких притч в стиле мифов, истории и духовных
историй для видео. Серия: "{series}". Тема: "{topic}"
Нужно ровно {n_scenes} сцен для ролика длительностью {duration_sec} секунд.

Верни СТРОГО валидный JSON-массив без markdown и без пояснений:
[
  {{"narration": "текст закадрового голоса для сцены (1 предложение, на английском)",
    "visual_prompt": "детальное описание кадра для художника: место, действие,
     освещение, настроение, без упоминания стиля рисовки"}}
]"""


def _parse_scenes_json(raw: str) -> list:
    raw = raw.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:]
    # Qwen иногда добавляет рассуждения до/после JSON — вырезаем по первой [ и последней ]
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Не нашёл JSON-массив в ответе модели:\n{raw[:500]}")
    return json.loads(raw[start:end + 1])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _generate_spec_qwen(prompt: str) -> list:
    """Локальная генерация через Ollama. Требует: ollama pull qwen2.5:32b (или указанную в QWEN_MODEL)."""
    result = subprocess.run(
        ["ollama", "run", QWEN_MODEL, prompt],
        capture_output=True, text=True, timeout=300, check=True,
    )
    return _parse_scenes_json(result.stdout)


def generate_spec(run_id: str, topic: str, series: str, duration_sec: int,
                   character_id: Optional[str], quality: str) -> Spec:
    n_scenes = max(6, duration_sec // 6)
    prompt = _spec_prompt(series, topic, n_scenes, duration_sec)

    print(f"[spec] генерирую через локальный Qwen ({QWEN_MODEL})")
    data = _generate_spec_qwen(prompt)

    scenes = [
        Scene(index=i, narration=item["narration"], visual_prompt=item["visual_prompt"])
        for i, item in enumerate(data)
    ]
    hero_indices = {0, len(scenes) // 2, len(scenes) - 1}
    for s in scenes:
        s.is_hero = s.index in hero_indices

    return Spec(
        run_id=run_id, topic=topic, series=series, duration_sec=duration_sec,
        character_id=character_id, quality=quality, scenes=scenes,
    )


# ============================================================
# Шаг 2. FLUX Image: кадры сцен (с привязкой к референсу персонажа)
# ============================================================
def generate_scene_images(spec: Spec):
    ref_image_path = None
    if spec.character_id:
        reg = load_character_registry()
        if spec.character_id in reg:
            ref_image_path = Path(reg[spec.character_id]["path"])
        else:
            print(f"⚠ character_id '{spec.character_id}' не найден в реестре — "
                  f"сцены сгенерятся без привязки к референсу персонажа")

    for scene in spec.scenes:
        if scene.status != "pending":
            print(f"[image] сцена {scene.index} уже готова (resume) — пропуск")
            continue

        full_prompt = (
            f"{scene.visual_prompt}{STYLE_SUFFIX}. "
            f"Featuring the same character shown in the reference image, "
            f"keep all facial features and identity identical."
            if ref_image_path else f"{scene.visual_prompt}{STYLE_SUFFIX}"
        )

        image = _generate_image(full_prompt, ref_image_path)
        out_path = SCENES / f"{spec.run_id}_scene_{scene.index:03d}.png"
        image.save(out_path)

        scene.image_path = str(out_path)
        scene.status = "image_done"
        save_state(spec)  # чекпоинт после каждой сцены — resume-safe
        print(f"[image] сцена {scene.index}/{len(spec.scenes)-1} готова -> {out_path}")


# ============================================================
# Шаг 3. Анимация: zoompan (дёшево) или Wan 2.2 (hero-сцены)
# ============================================================
def animate_zoompan(scene: Scene, seconds: float = 6.0) -> Path:
    out_path = SCENES / f"scene_{scene.index:03d}_anim.mp4"
    zoom_expr = "min(zoom+0.0015,1.15)"
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", scene.image_path,
        "-vf", f"zoompan=z='{zoom_expr}':d={int(seconds*25)}:s=1080x1920:fps=25",
        "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def animate_wan(scene: Scene, seconds: float = 5.0) -> Path:
    out_path = SCENES / f"scene_{scene.index:03d}_anim.mp4"
    cmd = [
        "python3", str(ROOT / "repos" / "Wan2.2" / "generate.py"),
        "--task", "i2v-A14B",
        "--ckpt_dir", str(MODELS / "wan2.2-i2v"),
        "--image", scene.image_path,
        "--prompt", scene.visual_prompt,
        "--frame_num", str(int(seconds * 16)),  # уточнить fps модели в её README
        "--save_file", str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def animate_scenes(spec: Spec):
    for scene in spec.scenes:
        if scene.status == "anim_done":
            print(f"[anim] сцена {scene.index} уже анимирована (resume) — пропуск")
            continue

        if scene.is_hero:
            print(f"[Wan] анимирую hero-сцену {scene.index}")
            scene.video_path = str(animate_wan(scene))
        else:
            print(f"[ffmpeg] zoompan сцена {scene.index}")
            scene.video_path = str(animate_zoompan(scene))

        scene.status = "anim_done"
        save_state(spec)


# ============================================================
# Шаг 4. Intro/outro серии
# ============================================================
def generate_intro_outro(spec: Spec) -> tuple[Path, Path]:
    template = get_series_template(spec.series)

    intro_prompt = (
        f"{template['intro_prompt']}, text overlay space for title '{spec.topic}'"
        f"{STYLE_SUFFIX}"
    )
    intro_image = _generate_image(intro_prompt)
    intro_img_path = SCENES / f"{spec.run_id}_intro.png"
    intro_image.save(intro_img_path)

    intro_video = SCENES / f"{spec.run_id}_intro.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(intro_img_path),
        "-vf", "zoompan=z='min(zoom+0.002,1.1)':d=75:s=1080x1920:fps=25",
        "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(intro_video)
    ], check=True, capture_output=True)

    # outro — переиспользуем intro-кадр с затемнением как простой, дешёвый вариант
    outro_video = SCENES / f"{spec.run_id}_outro.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(intro_img_path),
        "-vf", "fade=t=in:st=0:d=1,zoompan=z='1.05':d=50:s=1080x1920:fps=25",
        "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(outro_video)
    ], check=True, capture_output=True)

    return intro_video, outro_video


# ============================================================
# Шаг 5. XTTS-v2: озвучка
# ============================================================
def synthesize_voice(spec: Spec) -> Path:
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    full_text = " ".join(s.narration for s in spec.scenes)
    out_path = OUTPUT / f"{spec.run_id}_voice.wav"

    tts.tts_to_file(
        text=full_text,
        speaker_wav=str(VOICE_SAMPLE),
        language="en",
        file_path=str(out_path),
    )
    return out_path


# ============================================================
# Шаг 6. faster-whisper: субтитры со словными таймкодами
# ============================================================
def generate_subtitles(spec: Spec, audio_path: Path) -> Path:
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)

    template = get_series_template(spec.series)
    accent = template["accent_color"].lstrip("#")
    # ASS цвет в формате &HBBGGRR& — переворачиваем из hex RRGGBB
    ass_color = f"&H00{accent[4:6]}{accent[2:4]}{accent[0:2]}&"

    ass_path = OUTPUT / f"{spec.run_id}_subs.ass"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Default,Montserrat,64,&H00FFFFFF,{ass_color},&H00000000,&H80000000,1,3,1,2,60,60,120

[Events]
Format: Layer, Start, End, Style, Text
"""
    lines = [header]

    def ts(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        cs = int((s - int(s)) * 100)
        return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"

    for seg in segments:
        for word in seg.words:
            lines.append(
                f"Dialogue: 0,{ts(word.start)},{ts(word.end)},Default,"
                f"{{\\c{ass_color}}}{word.word}{{\\c&HFFFFFF&}}\n"
            )

    ass_path.write_text("".join(lines), encoding="utf-8")
    return ass_path


# ============================================================
# Шаг 7. Финальная сборка
# ============================================================
def assemble_final(spec: Spec, voice_path: Path, subs_path: Path,
                    intro: Path, outro: Path) -> Path:
    concat_list = OUTPUT / f"{spec.run_id}_concat.txt"
    with open(concat_list, "w") as f:
        f.write(f"file '{intro.resolve()}'\n")
        for scene in spec.scenes:
            f.write(f"file '{Path(scene.video_path).resolve()}'\n")
        f.write(f"file '{outro.resolve()}'\n")

    silent_video = OUTPUT / f"{spec.run_id}_silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(silent_video)
    ], check=True, capture_output=True)

    safe_topic = "".join(c if c.isalnum() else "_" for c in spec.topic)[:30]
    final_path = OUTPUT / f"final_{spec.series}_{safe_topic}_{spec.run_id}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(silent_video), "-i", str(voice_path),
        "-vf", f"ass={subs_path}",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(final_path)
    ], check=True, capture_output=True)

    return final_path


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--series", default="original",
                         choices=list(DEFAULT_SERIES_TEMPLATES.keys()))
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--quality", default="fast", choices=["fast", "max"],
                         help="fast=28 steps (быстрее), max=50 steps+выше guidance (лучше консистентность/детализация)")
    parser.add_argument("--character", default=None,
                         help="character_id из реестра; если не в реестре — используй --new-character")
    parser.add_argument("--new-character", default=None,
                         help="описание персонажа для первичной генерации reference sheet")
    parser.add_argument("--run-id", default=None,
                         help="для resume: тот же run-id продолжит с места обрыва")
    args = parser.parse_args()

    run_id = args.run_id or f"{int(time.time())}"

    global IMAGE_STEPS, IMAGE_GUIDANCE
    if args.quality == "max":
        IMAGE_STEPS, IMAGE_GUIDANCE = 50, 4.0
    else:
        IMAGE_STEPS, IMAGE_GUIDANCE = 28, 3.5

    if args.new_character and args.character:
        create_character(args.character, args.new_character)

    spec = load_state(run_id)
    if spec is None:
        print(f"=== [{run_id}] 1/7 Спек-документ (Qwen) ===")
        spec = generate_spec(run_id, args.topic, args.series, args.duration,
                              args.character, args.quality)
        save_state(spec)
    else:
        print(f"=== [{run_id}] Найден незавершённый прогон — резюмирую ===")

    print(f"=== [{run_id}] 2/7 Кадры сцен (FLUX) ===")
    generate_scene_images(spec)

    print(f"=== [{run_id}] 3/7 Анимация (zoompan + Wan для hero-сцен) ===")
    animate_scenes(spec)

    print(f"=== [{run_id}] 4/7 Intro/outro серии '{spec.series}' ===")
    intro, outro = generate_intro_outro(spec)

    print(f"=== [{run_id}] 5/7 Озвучка (XTTS-v2) ===")
    voice_path = synthesize_voice(spec)

    print(f"=== [{run_id}] 6/7 Субтитры (faster-whisper) ===")
    subs_path = generate_subtitles(spec, voice_path)

    print(f"=== [{run_id}] 7/7 Финальная сборка ===")
    final_path = assemble_final(spec, voice_path, subs_path, intro, outro)

    print(f"\n✅ ГОТОВО: {final_path}")


if __name__ == "__main__":
    main()
