# AI-конвейер рисованных роликов v3 — полностью автономный, без внешних API

## Что изменилось против v2

Убрали Gemini/Nano Banana полностью — нет квоты, работаем целиком на своём
железе. Вернули локальный FLUX.1-dev + style-LoRA + PuLID/IP-Adapter для
картинок и консистентности персонажа, встроив это в архитектуру v2
(реестр персонажей, серии, resume, локальный Qwen для сценариев).

**Всё локально на L4, без единого внешнего API:**
- Ollama + Qwen2.5 — сценарий/спек-документ
- FLUX.1-dev + style-LoRA + PuLID/IP-Adapter — кадры сцен, консистентность
  персонажа через reference-image
- Wan 2.2 — анимация hero-сцен
- ffmpeg — zoompan для обычных сцен + вся сборка
- XTTS-v2 — озвучка
- faster-whisper — субтитры

## Порядок действий

1. `cd aipipe2 && bash setup.sh 2>&1 | tee setup.log`
2. Впиши в `config.env`:
   - `HF_TOKEN` (обязательно приняты условия лицензии FLUX.1-dev на HF)
   - `STYLE_LORA_URL` — ссылка на .safetensors с Civitai под нужный
     painterly-стиль (протестируй 2-3 кандидата на 5-10 генерациях,
     оставь одну)
3. Положи `assets/voice_sample.wav` (10-20 сек чистой речи).
4. Скопируй `characters/registry.example.json` → `characters/registry.json`,
   `series_templates/templates.example.json` → `series_templates/templates.json`.
5. `source venv/bin/activate && source config.env`

## Первый прогон

```bash
python3 pipeline.py \
  --topic "Икар и крылья из воска" \
  --series mythology \
  --duration 90 \
  --quality fast \
  --character icarus_default \
  --new-character "young man, curly dark brown hair, olive skin, athletic build, simple linen tunic, Ancient Greek aesthetic"
```

Дальше для той же серии/персонажа `--new-character` не нужен:

```bash
python3 pipeline.py --topic "Дедал и лабиринт" --series mythology \
  --duration 90 --quality fast --character icarus_default
```

## Resume после сбоя

Каждая сцена чекпоинтится в `state/<run_id>.json` сразу после генерации
картинки и после анимации:

```bash
python3 pipeline.py --topic "..." --series mythology --duration 90 \
  --quality fast --character icarus_default --run-id 1234567890
```

## Качество: fast vs max

`--quality fast` — 28 шагов инференса FLUX, guidance 3.5. Быстрее.
`--quality max` — 50 шагов, guidance 4.0. Медленнее, но выше детализация
и стабильнее держит консистентность — используй для флагманских тем или
если `fast` даёт заметный разнобой в лице персонажа между сценами.

## Известные места для правки на месте

- **PuLID-FLUX**: репозиторий активно меняется, имя модуля/класса в
  `get_flux_pipeline()` может не совпасть — открой `repos/PuLID/README.md`
  на месте и поправь импорт. Если не взлетает быстро — код сам упадёт в
  fallback на IP-Adapter (см. try/except), можно просто продолжать на нём.
- **Wan2.2 CLI-флаги**: `generate.py --help` в свежескачанном репо покажет
  актуальные аргументы, если `--frame_num`/`--task` не совпадут.
- **Qwen VRAM-конфликт**: `qwen2.5:32b` грузится в момент шага 1 (спек),
  FLUX — на шаге 2, Wan — на шаге 3. Они не работают одновременно в памяти,
  но если увидишь OOM на переходе между шагами — смени `QWEN_MODEL` на
  `qwen2.5:14b` в `config.env`.

## Первое, что нужно сделать ДО реальных прогонов

Custom disk image (не просто снапшот) — новая VM поднимается за 3-5 минут
вместо часов установки:

```bash
gcloud compute images create ai-pipeline-base \
  --source-disk=YOUR_DISK_NAME --source-disk-zone=YOUR_ZONE \
  --family=ai-pipeline
```

Веса (FLUX + Wan2.2, десятки GB) — на отдельный persistent disk:

```bash
gcloud compute disks create ai-pipeline-models --size=200GB --zone=YOUR_ZONE
```

## Экономика (ориентир)

Себестоимость — только аренда GPU-времени, никаких API-расходов вообще:
- Short (60-90с, гибрид zoompan + 1-3 hero-сцены на Wan): ~$0.2-0.5/ролик
- Long (10-15 мин, тот же гибрид): ~$1.5-3/ролик
- `--quality max` увеличивает время генерации FLUX в ~1.5-2 раза за счёт
  роста числа шагов — цена растёт пропорционально времени аренды
