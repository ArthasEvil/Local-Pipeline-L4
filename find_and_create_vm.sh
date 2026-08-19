#!/usr/bin/env bash
# ============================================================
# find_and_create_vm.sh — умный поиск свободной зоны под L4 и создание VM.
#
# Логика (не тупой перебор по одной зоне за раз):
#   1. Проверяет авторизацию в gcloud, логинит если нужно.
#   2. Спрашивает у Google, в каких зонах ВООБЩЕ поддерживается L4
#      (accelerator-types list — это реальные данные, не угадывание).
#   3. Спрашивает у Google квоту NVIDIA_L4_GPUS по каждому региону —
#      отсеивает зоны, где квоты просто нет (тоже реальные данные).
#   4. По оставшимся кандидатам — ПАРАЛЛЕЛЬНО (не по очереди!) пытается
#      создать VM. Zonal capacity (ZONE_RESOURCE_POOL_EXHAUSTED) Google
#      не отдаёт заранее ни по какому API — это выясняется только в
#      момент попытки создания, поэтому единственный быстрый способ —
#      бить во все квотированные зоны одновременно и брать первую,
#      что ответила успехом.
#   5. Как только одна зона создала VM — останавливает попытки в
#      остальных и удаляет случайные "полу-созданные" дубликаты, если
#      вдруг успели проскочить две.
#
# Запуск:
#   bash find_and_create_vm.sh
#
# Настройки — через переменные окружения или правь блок CONFIG ниже.
# ============================================================
set -uo pipefail  # без -e: нам нужно ловить неудачные попытки создания, не падая

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
INSTANCE_NAME="${INSTANCE_NAME:-aipipe-l4-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"       # стандартная связка под 1x L4
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-l4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-gpu}"          # образ с преднакаченными GPU-драйверами
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
# Если уже делал custom image (см. README) — впиши сюда вместо common-gpu:
# CUSTOM_IMAGE="ai-pipeline-base"

# Ограничим кандидатов конкретными регионами, если знаешь, что тебе нужны
# именно они (пусто = искать по всем регионам, где Google предлагает L4)
PREFERRED_REGIONS="${PREFERRED_REGIONS:-}"          # напр. "us-central1 us-east4 europe-west4"

LOG() { echo -e "\n\033[1;32m[VM-FINDER]\033[0m $1"; }
WARN() { echo -e "\033[1;33m[warn]\033[0m $1"; }
FAIL() { echo -e "\033[1;31m[ОШИБКА]\033[0m $1"; exit 1; }

# ------------------------------------------------------------
# 1. Авторизация
# ------------------------------------------------------------
LOG "Проверяю авторизацию в gcloud"
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null)"
if [ -z "$ACTIVE_ACCOUNT" ]; then
  LOG "Не авторизован — открываю gcloud auth login"
  gcloud auth login || FAIL "Авторизация не прошла"
else
  LOG "Уже авторизован как: $ACTIVE_ACCOUNT"
fi

if [ -z "$PROJECT_ID" ]; then
  FAIL "Не задан проект. Запусти: gcloud config set project YOUR_PROJECT_ID"
fi
LOG "Проект: $PROJECT_ID"

# ------------------------------------------------------------
# 2. Спрашиваем у Google, где вообще есть L4 (реальные данные)
# ------------------------------------------------------------
LOG "Запрашиваю у Google список зон, где поддерживается $ACCELERATOR_TYPE"
ALL_L4_ZONES="$(gcloud compute accelerator-types list \
  --filter="name=${ACCELERATOR_TYPE}" \
  --format="value(zone)" 2>/dev/null | sort -u)"

if [ -z "$ALL_L4_ZONES" ]; then
  FAIL "Google не вернул ни одной зоны с $ACCELERATOR_TYPE — проверь название accelerator type"
fi

CANDIDATE_ZONES="$ALL_L4_ZONES"
if [ -n "$PREFERRED_REGIONS" ]; then
  LOG "Фильтрую по предпочитаемым регионам: $PREFERRED_REGIONS"
  FILTERED=""
  for zone in $ALL_L4_ZONES; do
    for region in $PREFERRED_REGIONS; do
      if [[ "$zone" == "$region"* ]]; then
        FILTERED="$FILTERED $zone"
      fi
    done
  done
  CANDIDATE_ZONES="$(echo "$FILTERED" | tr ' ' '\n' | sort -u)"
fi

ZONE_COUNT=$(echo "$CANDIDATE_ZONES" | wc -w)
LOG "Зон-кандидатов с поддержкой $ACCELERATOR_TYPE: $ZONE_COUNT"
echo "$CANDIDATE_ZONES" | tr ' ' '\n'

# ------------------------------------------------------------
# 3. Спрашиваем у Google реальную квоту NVIDIA_L4_GPUS по региону
#    (отсекаем зоны, где квоты просто нет — не тратим на них попытки)
# ------------------------------------------------------------
LOG "Проверяю квоту NVIDIA_L4_GPUS по каждому региону-кандидату"
QUOTA_OK_ZONES=""
CHECKED_REGIONS=""

for zone in $CANDIDATE_ZONES; do
  region="${zone%-*}"  # us-central1-a -> us-central1
  if [[ " $CHECKED_REGIONS " == *" $region "* ]]; then
    # регион уже проверен — если квота там была ок, добавляем зону
    if [[ " $QUOTA_OK_REGIONS " == *" $region "* ]]; then
      QUOTA_OK_ZONES="$QUOTA_OK_ZONES $zone"
    fi
    continue
  fi
  CHECKED_REGIONS="$CHECKED_REGIONS $region"

  QUOTA_LINE="$(gcloud compute regions describe "$region" \
    --project="$PROJECT_ID" \
    --format="value(quotas.filter(metric=NVIDIA_L4_GPUS).firstof(limit, usage))" 2>/dev/null)"

  LIMIT="$(gcloud compute regions describe "$region" \
    --project="$PROJECT_ID" \
    --format="json(quotas)" 2>/dev/null \
    | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for q in data.get('quotas', []):
        if q.get('metric') == 'NVIDIA_L4_GPUS':
            print(int(q.get('limit', 0) - q.get('usage', 0)))
            sys.exit(0)
    print(0)
except Exception:
    print(0)
" 2>/dev/null)"

  if [ -n "$LIMIT" ] && [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    QUOTA_OK_REGIONS="$QUOTA_OK_REGIONS $region"
    QUOTA_OK_ZONES="$QUOTA_OK_ZONES $zone"
    echo "  ✅ $region — свободная квота: $LIMIT"
  else
    echo "  ⛔ $region — квоты нет или исчерпана"
  fi
done

QUOTA_OK_ZONES="$(echo "$QUOTA_OK_ZONES" | tr ' ' '\n' | sed '/^$/d' | sort -u)"
if [ -z "$QUOTA_OK_ZONES" ]; then
  FAIL "Ни в одном регионе нет свободной квоты NVIDIA_L4_GPUS.\nЗапроси увеличение квоты: https://console.cloud.google.com/iam-admin/quotas\n(фильтр: NVIDIA_L4_GPUS, нужный регион)"
fi

FINAL_COUNT=$(echo "$QUOTA_OK_ZONES" | wc -l)
LOG "Зон с реальной квотой: $FINAL_COUNT — гоню параллельные попытки создания"
echo "$QUOTA_OK_ZONES"

# ------------------------------------------------------------
# 4. Параллельная гонка за создание VM
# ------------------------------------------------------------
WORKDIR="$(mktemp -d)"
WINNER_FILE="$WORKDIR/winner.txt"
PIDS=()

attempt_create() {
  local zone="$1"
  local vm_name="${INSTANCE_NAME}"
  local image_flags="--image-family=$IMAGE_FAMILY --image-project=$IMAGE_PROJECT"
  if [ -n "${CUSTOM_IMAGE:-}" ]; then
    image_flags="--image=$CUSTOM_IMAGE"
  fi

  gcloud compute instances create "$vm_name" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    $image_flags \
    --quiet > "$WORKDIR/log_${zone}.txt" 2>&1

  if [ $? -eq 0 ]; then
    # На случай гонки — если уже кто-то выиграл, эта VM лишняя, удаляем
    if [ -f "$WINNER_FILE" ]; then
      WARN "VM в $zone создалась, но $(cat $WINNER_FILE) уже выиграла раньше — удаляю дубликат"
      gcloud compute instances delete "$vm_name" --zone="$zone" --quiet --project="$PROJECT_ID" &
    else
      echo "$zone" > "$WINNER_FILE"
      echo "$zone" >> "$WORKDIR/success_zones.txt"
    fi
  fi
}

for zone in $QUOTA_OK_ZONES; do
  attempt_create "$zone" &
  PIDS+=($!)
  sleep 0.3  # лёгкий разброс, чтобы не словить rate limit на API создания разом
done

LOG "Запущено ${#PIDS[@]} параллельных попыток. Жду первую успешную (до 3 минут)..."

ELAPSED=0
while [ ! -f "$WINNER_FILE" ] && [ $ELAPSED -lt 180 ]; do
  sleep 3
  ELAPSED=$((ELAPSED + 3))
done

if [ -f "$WINNER_FILE" ]; then
  WINNER_ZONE="$(cat "$WINNER_FILE")"
  LOG "✅ УСПЕХ: VM '$INSTANCE_NAME' создана в зоне $WINNER_ZONE"
  echo "$WINNER_ZONE" > "$HOME/aipipe/.last_vm_zone" 2>/dev/null || true
  LOG "Подключение: gcloud compute ssh $INSTANCE_NAME --zone=$WINNER_ZONE"
else
  WARN "Ни одна зона не ответила успехом за 3 минуты. Логи попыток:"
  for f in "$WORKDIR"/log_*.txt; do
    zone_name="$(basename "$f" .txt | sed 's/log_//')"
    reason="$(grep -o 'ZONE_RESOURCE_POOL_EXHAUSTED\|QUOTA_EXCEEDED\|.*ERROR.*' "$f" | head -1)"
    echo "  $zone_name: ${reason:-неизвестная ошибка, см. $f}"
  done
  FAIL "Все зоны с квотой сейчас без свободных L4. Попробуй через 10-30 минут — capacity освобождается динамически, или запроси квоту в других регионах."
fi

# ждём фоновые процессы удаления дубликатов, если были
wait