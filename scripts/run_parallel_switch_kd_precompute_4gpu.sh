#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh [--dry-run] [--clean-outputs] <stage>

Stages:
  teacher-precompute unified teacher precompute
  switch-logits
  all

Examples:
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh teacher-precompute
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh switch-logits
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh all
  bash scripts/run_parallel_switch_kd_precompute_4gpu.sh --dry-run all
  CLEAN_OUTPUTS=1 bash scripts/run_parallel_switch_kd_precompute_4gpu.sh teacher-precompute

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
  teacher-precompute|switch-logits|all)
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
BASE_CONFIG="${BASE_CONFIG:-configs/parsing_switch_kd.yaml}"
GENERATED_CONFIG_DIR="configs/generated"
export BASE_CONFIG NUM_SHARDS GENERATED_CONFIG_DIR

eval "$(
  python - <<'PY'
from __future__ import annotations

import os
import shlex
from pathlib import Path

import yaml

base_config_path = Path(os.environ["BASE_CONFIG"])
if not base_config_path.exists():
    raise SystemExit(f"ERROR: base config not found: {base_config_path}")

with base_config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

options = config.get("options", {})
data = config.get("data", {})
task_name = options.get("task_name")
quality = options.get("quality")
teacher_quantization = options.get("teacher_quantization")
student_quantization = options.get("student_quantization")

required = {
    "options.task_name": task_name,
    "options.quality": quality,
    "options.teacher_quantization": teacher_quantization,
    "options.student_quantization": student_quantization,
    "data.manifest_path": data.get("manifest_path"),
    "data.label_path": data.get("label_path"),
    "data.switch_logits_path": data.get("switch_logits_path"),
}
missing = [name for name, value in required.items() if value in (None, "")]
if missing:
    raise SystemExit(f"ERROR: missing required config values in {base_config_path}: {', '.join(missing)}")

label_profile = f"{quality}_{teacher_quantization}"
response_profile = f"{quality}_{teacher_quantization}_student_{student_quantization}"
profile_slug = f"{task_name}_{response_profile}"

replacements = {
    "task_name": task_name,
    "quality": quality,
    "teacher_quantization": teacher_quantization,
    "student_quantization": student_quantization,
    "label_profile": label_profile,
    "response_profile": response_profile,
}

def expand(value: str) -> str:
    return value.format(**replacements)

manifest_path = expand(str(data["manifest_path"]))
final_label_path = expand(str(data["label_path"]))
final_switch_logits_path = expand(str(data["switch_logits_path"]))
shard_dir = f"outputs/switch-kd/shards/{profile_slug}"

exports = {
    "TASK_NAME": task_name,
    "QUALITY": quality,
    "TEACHER_QUANTIZATION": teacher_quantization,
    "STUDENT_QUANTIZATION": student_quantization,
    "LABEL_PROFILE": label_profile,
    "RESPONSE_PROFILE": response_profile,
    "MANIFEST_PATH": manifest_path,
    "FINAL_LABEL_PATH": final_label_path,
    "FINAL_SWITCH_LOGITS_PATH": final_switch_logits_path,
    "PROFILE_SLUG": profile_slug,
    "SHARD_DIR": shard_dir,
}

for key, value in exports.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

export MANIFEST_PATH FINAL_LABEL_PATH FINAL_SWITCH_LOGITS_PATH PROFILE_SLUG SHARD_DIR

mkdir -p "${SHARD_DIR}" "${GENERATED_CONFIG_DIR}"

split_manifest() {
  python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"])
shard_dir = Path(os.environ["SHARD_DIR"])
num_shards = int(os.environ["NUM_SHARDS"])

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
  echo "=== clean stale ${stage} shard outputs ==="
  for gpu_index in 0 1 2 3; do
    local shard_path=""
    if [[ "${stage}" == "teacher-precompute" ]]; then
      shard_path="${SHARD_DIR}/parsing_teacher_labels_shard${gpu_index}.jsonl"
    elif [[ "${stage}" == "switch-logits" ]]; then
      shard_path="${SHARD_DIR}/parsing_switch_logits_shard${gpu_index}.jsonl"
    fi
    if [[ -n "${shard_path}" && -f "${shard_path}" ]]; then
      echo "Removing ${shard_path}"
      rm -f "${shard_path}"
    fi
  done
}

generate_configs() {
  local stage="$1"
  STAGE_NAME="${stage}" python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import yaml

stage = os.environ["STAGE_NAME"]
stage_slug = stage.replace("-", "_")

base_config_path = Path(os.environ["BASE_CONFIG"])
generated_dir = Path(os.environ["GENERATED_CONFIG_DIR"])
shard_dir = Path(os.environ["SHARD_DIR"])
profile_slug = os.environ["PROFILE_SLUG"]

with base_config_path.open("r", encoding="utf-8") as handle:
    base_config = yaml.safe_load(handle)

distillation = base_config.get("distillation", {})
if distillation.get("method") != "switch_kd":
    raise SystemExit(
        f"ERROR: base config must use distillation.method=switch_kd: {base_config_path}"
    )
if distillation.get("teacher_logits") is not True:
    raise SystemExit(
        f"ERROR: base config must use distillation.teacher_logits=true: {base_config_path}"
    )

for gpu_index in range(int(os.environ["NUM_SHARDS"])):
    config_data = yaml.safe_load(yaml.safe_dump(base_config, sort_keys=False))
    config_data["data"]["manifest_path"] = str(
        shard_dir / f"parsing_manifest_shard{gpu_index}.jsonl"
    )
    config_data["data"]["label_path"] = str(
        shard_dir / f"parsing_teacher_labels_shard{gpu_index}.jsonl"
    )
    config_data.get("data", {}).pop("teacher_logits_path", None)
    if stage == "switch-logits":
        config_data["data"]["switch_logits_path"] = str(
            shard_dir / f"parsing_switch_logits_shard{gpu_index}.jsonl"
        )
        config_data["distillation"]["student_visual_cache_dir"] = str(
            shard_dir / f"student_visual_cache_shard{gpu_index}"
        )
    output_path = generated_dir / f"parsing_switch_kd_{profile_slug}_{stage_slug}_gpu{gpu_index}.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False, allow_unicode=True)
    print(f"[generate-config] {output_path}")
    print(
        "[generate-config-detail] "
        f"base_config={base_config_path} "
        f"generated_config={output_path} "
        f"gpu={gpu_index} distillation.method={config_data.get('distillation', {}).get('method')} "
        f"distillation.teacher_logits={config_data.get('distillation', {}).get('teacher_logits')} "
        f"label_path={config_data.get('data', {}).get('label_path')} "
        f"teacher_logits_path=deprecated "
        f"teacher_logits_field={config_data.get('distillation', {}).get('teacher_logits_field')} "
        f"unified_teacher_precompute_enabled=true "
        f"canonical_teacher_output_path=label_path "
        f"switch_kd.visual_switch.mode={config_data.get('distillation', {}).get('switch_kd', {}).get('visual_switch', {}).get('mode')} "
        f"switch_kd.visual_switch.teacher_projector={config_data.get('distillation', {}).get('switch_kd', {}).get('visual_switch', {}).get('teacher_projector')} "
        f"switch_kd.visual_switch.allow_fallback_adapter={config_data.get('distillation', {}).get('switch_kd', {}).get('visual_switch', {}).get('allow_fallback_adapter')}"
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
shard_dir = Path(os.environ["SHARD_DIR"])

required: list[Path] = []
if stage == "switch-logits":
    required = []
    for gpu_index in range(int(os.environ["NUM_SHARDS"])):
        required.append(shard_dir / f"parsing_teacher_labels_shard{gpu_index}.jsonl")

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
  echo "=== teacher precompute dry-run ==="
  python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import yaml

base_config_path = Path(os.environ["BASE_CONFIG"])
with base_config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
distillation = config.get("distillation", {})
data = config.get("data", {})
teacher_logits = bool(distillation.get("teacher_logits", True))
label_path = data.get("label_path")
mode = "unified"
print(f"base config path: {base_config_path}")
print(f"manifest path: {os.environ['MANIFEST_PATH']}")
print(f"profile slug: {os.environ['PROFILE_SLUG']}")
print(f"shard dir: {os.environ['SHARD_DIR']}")
print(f"final label path: {os.environ['FINAL_LABEL_PATH']}")
print(f"final switch logits path: {os.environ['FINAL_SWITCH_LOGITS_PATH']}")
print(f"method: {distillation.get('method')}")
print(f"teacher_logits: {str(teacher_logits).lower()}")
print(f"canonical teacher output path is label_path: {label_path}")
print("unified teacher precompute enabled")
print(f"teacher output mode: {mode}")
visual_switch = distillation.get("switch_kd", {}).get("visual_switch", {})
print(f"Switch-KD visual-switch mode: {visual_switch.get('mode', 'paper')}")
print("T-Projector definition: teacher native projector / merger")
print("Visual-switch path: student visual encoder output -> teacher projector/merger -> teacher LLM")
fallback_adapter = visual_switch.get("allow_fallback_adapter", False)
print(f"Fallback adapter: {'enabled' if fallback_adapter else 'disabled'}")
PY
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

import os
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"])
shard_dir = Path(os.environ["SHARD_DIR"])
num_shards = int(os.environ["NUM_SHARDS"])

total = 0
with manifest_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            total += 1
print(f"{manifest_path}: {total}")

for gpu_index in range(num_shards):
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
    echo "  config=${GENERATED_CONFIG_DIR}/parsing_switch_kd_${PROFILE_SLUG}_${stage_slug}_gpu${gpu_index}.yaml"
    echo "  label_path=${SHARD_DIR}/parsing_teacher_labels_shard${gpu_index}.jsonl"
    echo "  canonical_teacher_output_path=label_path"
    echo "  unified teacher precompute enabled"
    echo "  teacher_logits_path=deprecated"
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
    echo "CUDA_VISIBLE_DEVICES=${gpu_index} python -m vlm_distill.cli ${stage} --config ${GENERATED_CONFIG_DIR}/parsing_switch_kd_${PROFILE_SLUG}_${stage_slug}_gpu${gpu_index}.yaml > ${SHARD_DIR}/${stage_slug}_gpu${gpu_index}.log 2>&1"
  done
}

render_stage_progress_snapshot() {
  local stage="$1"
  shift
  python - "$stage" "$@" <<'PY'
from __future__ import annotations

import sys
import os
from pathlib import Path

stage = sys.argv[1]
pids = [int(arg) for arg in sys.argv[2:]]
stage_slug = stage.replace("-", "_")
shard_dir = Path(os.environ["SHARD_DIR"])

if stage == "teacher-precompute":
    output_template = "parsing_teacher_labels_shard{gpu}.jsonl"
elif stage == "switch-logits":
    output_template = "parsing_switch_logits_shard{gpu}.jsonl"
else:
    raise SystemExit(f"ERROR: unsupported monitor stage: {stage}")

def count_non_empty_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0

def last_non_empty_line(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return "[missing]"
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return "[empty]"

def truncate(text: str, limit: int = 180) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

header = (
    f"[progress:{stage}] "
    "gpu pid shard_rows output_rows percent log_path last_non_empty_log_line"
)
print(header)
for gpu_index, pid in enumerate(pids):
    manifest_path = shard_dir / f"parsing_manifest_shard{gpu_index}.jsonl"
    output_path = shard_dir / output_template.format(gpu=gpu_index)
    log_path = shard_dir / f"{stage_slug}_gpu{gpu_index}.log"
    shard_rows = count_non_empty_lines(manifest_path)
    output_rows = count_non_empty_lines(output_path)
    percent = 0.0
    if shard_rows > 0:
        percent = min(100.0, (output_rows / shard_rows) * 100.0)
    last_line = truncate(last_non_empty_line(log_path))
    print(
        f"[progress:{stage}] "
        f"gpu={gpu_index} pid={pid} shard_rows={shard_rows} "
        f"output_rows={output_rows} percent={percent:.1f}% "
        f"log_path={log_path} last_non_empty_log_line={last_line}"
    )
PY
}

monitor_stage_progress_start() {
  local stage="$1"
  shift
  MONITOR_STAGE="${stage}"
  MONITOR_PIDS=("$@")
  MONITOR_STATE_DIR="$(mktemp -d "/tmp/switch-kd-${stage//[^[:alnum:]_]/_}.XXXXXX")"
  (
    set +e
    render_stage_progress_snapshot "${MONITOR_STAGE}" "${MONITOR_PIDS[@]}"
    while [[ ! -f "${MONITOR_STATE_DIR}/stop" ]]; do
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        [[ -f "${MONITOR_STATE_DIR}/stop" ]] && break
        sleep 1
      done
      [[ -f "${MONITOR_STATE_DIR}/stop" ]] && break
      render_stage_progress_snapshot "${MONITOR_STAGE}" "${MONITOR_PIDS[@]}"
    done
  ) &
  MONITOR_PID=$!
}

monitor_stage_progress_stop() {
  if [[ -n "${MONITOR_STATE_DIR:-}" ]]; then
    touch "${MONITOR_STATE_DIR}/stop"
  fi
  if [[ -n "${MONITOR_PID:-}" ]]; then
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MONITOR_STATE_DIR:-}" && -d "${MONITOR_STATE_DIR}" ]]; then
    rm -rf "${MONITOR_STATE_DIR}"
  fi
  MONITOR_PID=""
  MONITOR_STATE_DIR=""
  MONITOR_STAGE=""
  MONITOR_PIDS=()
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

  trap 'monitor_stage_progress_stop' EXIT

  for gpu_index in 0 1 2 3; do
    local config_path="${GENERATED_CONFIG_DIR}/parsing_switch_kd_${PROFILE_SLUG}_${stage_slug}_gpu${gpu_index}.yaml"
    local log_path="${SHARD_DIR}/${stage_slug}_gpu${gpu_index}.log"
    gpu_logs+=("${log_path}")
    echo "Launching stage=${stage} gpu=${gpu_index}: ${config_path} -> ${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu_index}" \
      python -m vlm_distill.cli "${stage}" --config "${config_path}" \
      > "${log_path}" 2>&1 &
    pids+=("$!")
  done

  monitor_stage_progress_start "${stage}" "${pids[@]}"

  for i in "${!pids[@]}"; do
    local pid="${pids[$i]}"
    local gpu_index="${i}"
    if ! wait "${pid}"; then
      echo "ERROR: stage=${stage} failed on GPU ${gpu_index}. See ${gpu_logs[$i]}"
      failed=1
    fi
  done

  monitor_stage_progress_stop
  trap - EXIT

  echo "=== final progress summary: ${stage} ==="
  render_stage_progress_snapshot "${stage}" "${pids[@]}"

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
shard_dir = Path(os.environ["SHARD_DIR"])
manifest_path = Path(os.environ["MANIFEST_PATH"])

if stage == "teacher-precompute":
    shard_template = "parsing_teacher_labels_shard{gpu}.jsonl"
    final_output_path = Path(os.environ["FINAL_LABEL_PATH"])
elif stage == "switch-logits":
    shard_template = "parsing_switch_logits_shard{gpu}.jsonl"
    final_output_path = Path(os.environ["FINAL_SWITCH_LOGITS_PATH"])
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

def teacher_logits_enabled() -> bool:
    try:
        import yaml
        with Path(os.environ["BASE_CONFIG"]).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return bool(config.get("distillation", {}).get("teacher_logits", True))
    except Exception:
        return True

for shard_index in range(int(os.environ["NUM_SHARDS"])):
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
            if stage == "teacher-precompute" and is_valid_logits_payload(row.get("teacher_logits")):
                valid_logits_by_shard[shard_index] = valid_logits_by_shard.get(shard_index, 0) + 1
            rows.append(row)
    if stage == "teacher-precompute" and teacher_logits_enabled() and valid_logits_by_shard.get(shard_index, 0) <= 0:
        raise SystemExit(
            f"ERROR: unified teacher precompute shard has zero valid teacher_logits rows: {shard_path} stage={stage}"
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

if stage == "teacher-precompute" and teacher_logits_enabled():
    logits_rows = [row for row in rows if is_valid_logits_payload(row.get("teacher_logits"))]
    if len(seen_ids) != len(rows):
        raise SystemExit("ERROR: merged teacher logits ids are not unique")
    answer_rows = [row for row in rows if row.get("teacher_answer")]
    if len(answer_rows) != len(rows):
        raise SystemExit(
            f"ERROR: merged teacher rows missing teacher_answer: valid={len(answer_rows)} total={len(rows)}"
        )
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
if stage == "teacher-precompute" and teacher_logits_enabled():
    print(f"Validated teacher_logits rows: {len(rows)}")
PY
}

run_all() {
  run_stage "teacher-precompute"
  run_stage "switch-logits"
}

if [[ "${REQUESTED_STAGE}" == "all" ]]; then
  run_all
else
  run_stage "${REQUESTED_STAGE}"
fi
