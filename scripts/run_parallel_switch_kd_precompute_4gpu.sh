#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh [--dry-run] [--clean-outputs] <stage>

Stages:
  label
  teacher-logits
  switch-logits
  all

Examples:
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh label
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh teacher-logits
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh switch-logits
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh all
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh --dry-run all
  CLEAN_OUTPUTS=1 bash scripts/run_parallel_switch_kd_precompute_4gpu.sh teacher-logits

What it does:
  - Changes directory to ~/vlm_distill/Switch-KD
  - Activates conda env `vlm_distill`
  - Splits outputs/switch-kd/parsing_manifest.jsonl into 4 shard manifests
  - Generates stage-specific shard configs under configs/generated/
  - Runs 4 parallel workers per stage on GPUs 0-3
  - Merges successful shard outputs back into the normal final JSONL paths

Notes:
  - Resume behavior is preserved per shard because every worker writes to its own shard file.
  - Existing shard outputs and logs are preserved for resume/debugging unless CLEAN_OUTPUTS=1 or --clean-outputs is set.
  - `--dry-run` writes shard manifests and generated configs, prints planned commands, and skips worker launch/merge.
EOF
}

DRY_RUN=0
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-0}"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --clean-outputs|--force)
      CLEAN_OUTPUTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ $# -eq 0 ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    REQUESTED_STAGE="all"
  else
    usage
    exit 1
  fi
elif [[ $# -eq 1 ]]; then
  REQUESTED_STAGE="$1"
else
  usage
  exit 1
fi
case "${REQUESTED_STAGE}" in
  label|teacher-logits|switch-logits|all)
    ;;
  *)
    echo "ERROR: unsupported stage: ${REQUESTED_STAGE}"
    usage
    exit 1
    ;;
esac

PROJECT_ROOT="${HOME}/vlm_distill/Switch-KD"
cd "${PROJECT_ROOT}"

CONDA_SH="${HOME}/miniforge3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: conda activation script not found: ${CONDA_SH}"
  exit 1
fi
source "${CONDA_SH}"
conda activate vlm_distill

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_SHARDS=4
BASE_CONFIG="configs/parsing_switch_kd.yaml"
MANIFEST_PATH="outputs/switch-kd/parsing_manifest.jsonl"
SHARD_DIR="outputs/switch-kd/shards"
GENERATED_CONFIG_DIR="configs/generated"

FINAL_LABEL_PATH="outputs/switch-kd/parsing_teacher_labels_480p_8bit.jsonl"
FINAL_TEACHER_LOGITS_PATH="outputs/switch-kd/parsing_teacher_logits_480p_8bit.jsonl"
FINAL_SWITCH_LOGITS_PATH="outputs/switch-kd/parsing_switch_logits_480p_8bit_student_4bit.jsonl"

mkdir -p "${SHARD_DIR}" "${GENERATED_CONFIG_DIR}"

split_manifest() {
  python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

manifest_path = Path("outputs/switch-kd/parsing_manifest.jsonl")
shard_dir = Path("outputs/switch-kd/shards")
num_shards = 4

if not manifest_path.exists():
    raise SystemExit(f"ERROR: manifest not found: {manifest_path}")

rows: list[dict] = []
seen_ids: set[str] = set()
with manifest_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"ERROR: failed to parse JSONL at {manifest_path}:{line_number}: {exc}"
            ) from exc
        sample_id = row.get("id")
        if sample_id is None:
            raise SystemExit(
                f"ERROR: missing id in manifest row at {manifest_path}:{line_number}"
            )
        sample_id_str = str(sample_id)
        if sample_id_str in seen_ids:
            raise SystemExit(f"ERROR: duplicate id detected in manifest: {sample_id_str}")
        seen_ids.add(sample_id_str)
        rows.append(row)

shards: list[list[dict]] = [[] for _ in range(num_shards)]
for index, row in enumerate(rows):
    shards[index % num_shards].append(row)

for shard_index, shard_rows in enumerate(shards):
    shard_path = shard_dir / f"parsing_manifest_shard{shard_index}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in shard_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[split-manifest] {shard_path} rows={len(shard_rows)}")

print(f"[split-manifest] total_rows={len(rows)}")
PY
}

clean_stage_outputs() {
  local stage="$1"
  if [[ "${CLEAN_OUTPUTS}" != "1" ]]; then
    return 0
  fi
  if [[ "${stage}" != "teacher-logits" ]]; then
    return 0
  fi
  echo "=== clean stale teacher logits shard outputs ==="
  for gpu_index in 0 1 2 3; do
    local shard_path="${SHARD_DIR}/parsing_teacher_logits_shard${gpu_index}.jsonl"
    if [[ -f "${shard_path}" ]]; then
      echo "Removing ${shard_path}"
      rm -f "${shard_path}"
    fi
  done
}

generate_configs() {
  local stage="$1"
  STAGE_NAME="${stage}" python - <<'PY'
from __future__ import annotations

from pathlib import Path

import yaml

stage = __import__("os").environ["STAGE_NAME"]
stage_slug = stage.replace("-", "_")

base_config_path = Path("configs/parsing_switch_kd.yaml")
generated_dir = Path("configs/generated")
shard_dir = Path("outputs/switch-kd/shards")

with base_config_path.open("r", encoding="utf-8") as handle:
    base_config = yaml.safe_load(handle)

for gpu_index in range(4):
    config_data = yaml.safe_load(yaml.safe_dump(base_config, sort_keys=False))
    config_data["data"]["manifest_path"] = str(
        shard_dir / f"parsing_manifest_shard{gpu_index}.jsonl"
    )
    config_data["data"]["label_path"] = str(
        shard_dir / f"parsing_teacher_labels_shard{gpu_index}.jsonl"
    )
    if stage in {"teacher-logits", "switch-logits"}:
        config_data["data"]["teacher_logits_path"] = str(
            shard_dir / f"parsing_teacher_logits_shard{gpu_index}.jsonl"
        )
    if stage == "switch-logits":
        config_data["data"]["switch_logits_path"] = str(
            shard_dir / f"parsing_switch_logits_shard{gpu_index}.jsonl"
        )
        config_data["distillation"]["student_visual_cache_dir"] = str(
            shard_dir / f"student_visual_cache_shard{gpu_index}"
        )
    output_path = generated_dir / f"parsing_switch_kd_{stage_slug}_gpu{gpu_index}.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False, allow_unicode=True)
    print(f"[generate-config] {output_path}")
    if stage == "teacher-logits":
        print(
            "[generate-config-detail] "
            f"gpu={gpu_index} distillation.method={config_data.get('distillation', {}).get('method')} "
            f"teacher_logits_path={config_data.get('data', {}).get('teacher_logits_path')} "
            f"teacher_logits_field={config_data.get('distillation', {}).get('teacher_logits_field')}"
        )
PY
}

verify_stage_inputs() {
  local stage="$1"
  STAGE_NAME="${stage}" python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

stage = os.environ["STAGE_NAME"]
shard_dir = Path("outputs/switch-kd/shards")

required: list[Path] = []
if stage == "teacher-logits":
    required = [
        shard_dir / f"parsing_teacher_labels_shard{gpu_index}.jsonl"
        for gpu_index in range(4)
    ]
elif stage == "switch-logits":
    required = []
    for gpu_index in range(4):
        required.append(shard_dir / f"parsing_teacher_labels_shard{gpu_index}.jsonl")
        required.append(shard_dir / f"parsing_teacher_logits_shard{gpu_index}.jsonl")

missing = [path for path in required if not path.exists()]
if missing:
    for path in missing:
        print(f"ERROR: missing required input file: {path}")
    raise SystemExit(1)
PY
}

print_safety_checks() {
  local stage="$1"
  local stage_slug="${stage//-/_}"

  echo "=== stage ==="
  echo "${stage}"
  echo "=== current working directory ==="
  pwd
  echo "=== active python path ==="
  python - <<'PY'
import sys
print(sys.executable)
PY
  echo "=== nvidia-smi ==="
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found"
  elif ! nvidia-smi; then
    echo "WARNING: nvidia-smi returned non-zero status"
  fi
  echo "=== shard manifest counts ==="
  python - <<'PY'
from __future__ import annotations

from pathlib import Path

manifest_path = Path("outputs/switch-kd/parsing_manifest.jsonl")
shard_dir = Path("outputs/switch-kd/shards")

total = 0
with manifest_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            total += 1
print(f"{manifest_path}: {total}")

for gpu_index in range(4):
    shard_path = shard_dir / f"parsing_manifest_shard{gpu_index}.jsonl"
    count = 0
    with shard_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    print(f"{shard_path}: {count}")
PY
  echo "=== shard input/output/config paths ==="
  for gpu_index in 0 1 2 3; do
    echo "gpu=${gpu_index}"
    echo "  manifest=${SHARD_DIR}/parsing_manifest_shard${gpu_index}.jsonl"
    echo "  config=${GENERATED_CONFIG_DIR}/parsing_switch_kd_${stage_slug}_gpu${gpu_index}.yaml"
    echo "  label=${SHARD_DIR}/parsing_teacher_labels_shard${gpu_index}.jsonl"
    if [[ "${stage}" == "teacher-logits" || "${stage}" == "switch-logits" ]]; then
      echo "  teacher_logits=${SHARD_DIR}/parsing_teacher_logits_shard${gpu_index}.jsonl"
      if [[ -f "${SHARD_DIR}/parsing_teacher_logits_shard${gpu_index}.jsonl" ]]; then
        echo "  WARNING: existing teacher_logits shard output will be resumed unless invalid rows are recomputed or CLEAN_OUTPUTS=1 is set"
      fi
    fi
    if [[ "${stage}" == "switch-logits" ]]; then
      echo "  switch_logits=${SHARD_DIR}/parsing_switch_logits_shard${gpu_index}.jsonl"
      echo "  student_visual_cache_dir=${SHARD_DIR}/student_visual_cache_shard${gpu_index}"
    fi
  done
}

print_stage_commands() {
  local stage="$1"
  local stage_slug="${stage//-/_}"

  echo "=== planned commands ==="
  for gpu_index in 0 1 2 3; do
    echo "CUDA_VISIBLE_DEVICES=${gpu_index} python -m vlm_distill.cli ${stage} --config ${GENERATED_CONFIG_DIR}/parsing_switch_kd_${stage_slug}_gpu${gpu_index}.yaml > ${SHARD_DIR}/${stage_slug}_gpu${gpu_index}.log 2>&1"
  done
}

run_stage() {
  local stage="$1"
  local stage_slug="${stage//-/_}"
  local failed=0

  split_manifest
  clean_stage_outputs "${stage}"
  generate_configs "${stage}"
  print_safety_checks "${stage}"
  print_stage_commands "${stage}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] skipping ${stage} worker launch and merge"
    return 0
  fi

  verify_stage_inputs "${stage}"

  declare -a pids=()
  declare -a gpu_logs=()

  for gpu_index in 0 1 2 3; do
    local config_path="${GENERATED_CONFIG_DIR}/parsing_switch_kd_${stage_slug}_gpu${gpu_index}.yaml"
    local log_path="${SHARD_DIR}/${stage_slug}_gpu${gpu_index}.log"
    gpu_logs+=("${log_path}")
    echo "Launching stage=${stage} gpu=${gpu_index}: ${config_path} -> ${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu_index}" \
      python -m vlm_distill.cli "${stage}" --config "${config_path}" \
      > "${log_path}" 2>&1 &
    pids+=("$!")
  done

  for i in "${!pids[@]}"; do
    local pid="${pids[$i]}"
    local gpu_index="${i}"
    if ! wait "${pid}"; then
      echo "ERROR: stage=${stage} failed on GPU ${gpu_index}. See ${gpu_logs[$i]}"
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "ERROR: one or more ${stage} shard processes failed; merged output was not written."
    return 1
  fi

  merge_stage "${stage}"
}

merge_stage() {
  local stage="$1"
  STAGE_NAME="${stage}" python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

stage = os.environ["STAGE_NAME"]
shard_dir = Path("outputs/switch-kd/shards")
manifest_path = Path("outputs/switch-kd/parsing_manifest.jsonl")

if stage == "label":
    shard_template = "parsing_teacher_labels_shard{gpu}.jsonl"
    final_output_path = Path("outputs/switch-kd/parsing_teacher_labels_480p_8bit.jsonl")
elif stage == "teacher-logits":
    shard_template = "parsing_teacher_logits_shard{gpu}.jsonl"
    final_output_path = Path("outputs/switch-kd/parsing_teacher_logits_480p_8bit.jsonl")
elif stage == "switch-logits":
    shard_template = "parsing_switch_logits_shard{gpu}.jsonl"
    final_output_path = Path("outputs/switch-kd/parsing_switch_logits_480p_8bit_student_4bit.jsonl")
else:
    raise SystemExit(f"ERROR: unsupported merge stage: {stage}")

rows: list[dict] = []
seen_ids: set[str] = set()
valid_logits_by_shard: dict[int, int] = {}

def is_valid_logits_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if not all(key in payload for key in ("indices", "values", "vocab_size")):
        return False
    return bool(payload.get("indices")) and bool(payload.get("values"))

for shard_index in range(4):
    shard_path = shard_dir / shard_template.format(gpu=shard_index)
    if not shard_path.exists():
        raise SystemExit(f"ERROR: shard output file not found: {shard_path}")
    with shard_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"ERROR: failed to parse JSONL at {shard_path}:{line_number}: {exc}"
                ) from exc
            sample_id = row.get("id")
            if sample_id is None:
                raise SystemExit(
                    f"ERROR: missing id in merged input row from {shard_path}:{line_number}"
                )
            sample_id_str = str(sample_id)
            if sample_id_str in seen_ids:
                raise SystemExit(
                    f"ERROR: duplicate id detected during {stage} merge: {sample_id_str}"
                )
            seen_ids.add(sample_id_str)
            if stage == "teacher-logits" and is_valid_logits_payload(row.get("teacher_logits")):
                valid_logits_by_shard[shard_index] = valid_logits_by_shard.get(shard_index, 0) + 1
            rows.append(row)
    if stage == "teacher-logits" and valid_logits_by_shard.get(shard_index, 0) <= 0:
        raise SystemExit(
            f"ERROR: teacher-logits shard has zero valid teacher_logits rows: {shard_path}"
        )

def sort_key(row: dict) -> tuple[int, object]:
    value = row["id"]
    if isinstance(value, (int, float)):
        return (0, value)
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)

expected_count = 0
with manifest_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            expected_count += 1

rows.sort(key=sort_key)
if expected_count and len(rows) != expected_count:
    raise SystemExit(
        f"ERROR: merged row count mismatch for {stage}: merged={len(rows)} expected={expected_count}"
    )

if stage == "teacher-logits":
    logits_rows = [row for row in rows if is_valid_logits_payload(row.get("teacher_logits"))]
    if len(seen_ids) != len(rows):
        raise SystemExit("ERROR: merged teacher logits ids are not unique")
    if len(logits_rows) != len(rows):
        raise SystemExit(
            f"ERROR: merged teacher logits rows missing valid teacher_logits: valid={len(logits_rows)} total={len(rows)}"
        )
    if rows and not isinstance(rows[0].get("teacher_logits"), dict):
        raise SystemExit("ERROR: first merged teacher logits row does not contain teacher_logits dict")

final_output_path.parent.mkdir(parents=True, exist_ok=True)
with final_output_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Merged rows: {len(rows)}")
print(f"Expected rows: {expected_count}")
print(f"Merged output: {final_output_path}")
if stage == "teacher-logits":
    print(f"Validated teacher_logits rows: {len(rows)}")
PY
}

run_all() {
  run_stage "label"
  run_stage "teacher-logits"
  run_stage "switch-logits"
}

if [[ "${REQUESTED_STAGE}" == "all" ]]; then
  run_all
else
  run_stage "${REQUESTED_STAGE}"
fi
