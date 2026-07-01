# VLM Distillation Pipeline

Vision-Language Model Distillation Pipeline for GUI Automation Testing.

This project is designed for Android TV, mobile devices, tablets, in-vehicle infotainment systems, and other GUI-driven products. The goal is to generate high-quality teacher labels from large VLMs and distill them into smaller deployable models.

---

# Current Roadmap

Current milestone:

```text
Screen Parsing â†’ Auto Grounding Bootstrap Pipeline
```

Pipeline:

```text
Device Screenshot
        â†“
Screen Parsing Teacher
        â†“
UI Elements
        â†“
Grounding Teacher
        â†“
Bounding Boxes
        â†“
Student Distillation
```

---

# Installation

Install editable package:

```powershell
pip install -e .
```

Verify CLI:

```powershell
vlm-distill --help
```

---

# Supported Teacher Backends

## Hugging Face

Example local model:

```yaml
teacher:
  backend: hf
  model_name: D:/Models/Qwen2.5-VL-7B-Instruct

  device_map: auto
  torch_dtype: float16

  quantization: 4bit

  temperature: 0.0
  max_new_tokens: 256
  image_resize: original  # original, 720p, or 480p
```

Recommended for:

```text
RTX 4060Ti 16GB
Qwen2.5-VL-7B-Instruct
4bit NF4 quantization
```

---

## OpenAI Compatible

```yaml
teacher:
  backend: openai_compatible

  model_name: gpt-4o

  base_url: https://api.openai.com/v1
  api_key: YOUR_API_KEY
```

---

## Ollama

```yaml
teacher:
  backend: ollama

  model_name: llava:7b
```

---

# CLI Commands

## Create Manifest

Generate a manifest from image folders.

```powershell
vlm-distill create-manifest \
  --config configs/parsing_labeling.yaml \
```

or

```powershell
vlm-distill create-manifest \
  --config configs/grounding_test.yaml \
```

---

## Validate Manifest

```powershell
vlm-distill validate-manifest \
  --config configs/parsing_labeling.yaml
```

---

## Validate Teacher

```powershell
vlm-distill validate-teacher \
  --config configs/parsing_response_distillation.yaml
```

This reads the canonical teacher output at `data.label_path` and reports:

- `total_rows`
- `valid_json_rows`
- `schema_valid_rows`
- `string_list_rows`
- `answer_token_match_rows`
- `answer_token_mismatch_rows`
- `rows_with_teacher_logits` when `distillation.teacher_logits: true`
- `valid_teacher_logits_rows`
- `logits_length_match_rows`
- `logits_length_mismatch_rows`
- `full_sequence_logits_rows`
- `vocab_mismatch_rows`

---

## Generate Teacher Labels

```powershell
vlm-distill label \
  --config configs/parsing_labeling.yaml
```

`label` now runs unified teacher precompute. By default
`distillation.teacher_logits: true`, so Switch-KD configs write
`teacher_answer`, `teacher_tokens`, and answer-only `teacher_logits` from the
same final normalized answer to `data.label_path`. `data.label_path` is the
canonical teacher precompute output. `data.teacher_logits_path` is deprecated
and is not used by the main Switch-KD workflow.

Use `vlm-distill teacher-precompute --config ...` for the same stage with an
explicit name.

---

## Batch Predict With Student Or Merged Model

```powershell
vlm-distill predict \
  --config configs/parsing_response_distillation.yaml
```

This reads `data.manifest_path` and writes predictions to `data.prediction_path` when set, otherwise `data.distill_path`.

---

## Generate Switch-KD Visual Logits

```powershell
vlm-distill switch-logits \
  --config configs/switch_kd_4060ti.yaml
```

---

## Train Student

```powershell
vlm-distill train \
  --config configs/switch_kd_4060ti.yaml
```

Single-process single-GPU training with model sharding:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vlm_distill.cli train --config configs/parsing_switch_kd.yaml
```

Multi-GPU DDP training with Accelerate:

```bash
accelerate launch --num_processes 4 -m vlm_distill.cli train --config configs/parsing_switch_kd_ddp.yaml
```

Notes:

- `student.device_map: auto` means Hugging Face model sharding, not data parallel training.
- For multi-GPU DDP, use `student.device_map: null` so `from_pretrained()` does not shard the student model.
- `effective_batch = batch_size * gradient_accumulation_steps * num_gpus`
- Single GPU example: `batch_size=1`, `gradient_accumulation_steps=16`, `effective_batch=16`
- 4-GPU DDP example: `batch_size=1`, `gradient_accumulation_steps=4`, `effective_batch=1 * 4 * 4 = 16`

---

## Evaluate Student

```powershell
vlm-distill evaluate \
  --config configs/switch_kd_4060ti.yaml
```

---

## Evaluate Prediction JSONL

```powershell
vlm-distill evaluate-predictions \
  --config configs/parsing_response_distillation.yaml
```

This compares `data.prediction_path` against `data.eval_path` when set, otherwise `data.label_path` / `data.distill_path`.

---

# Screen Parsing Workflow

## Configuration

Example:

```yaml
data:
  image_dir: D:/TV_data/test_data

  output_dir: D:/TV_data/teacher_parsing

  manifest_path: D:/TV_data/teacher_parsing/parsing_manifest.jsonl

  distill_path: D:/TV_data/teacher_parsing/parsing_teacher_labels.jsonl

  eval_path: D:/TV_data/teacher_parsing/parsing_teacher_labels.jsonl

  image_root: .

  max_samples: 5
```

---

## Step 1

Generate manifest:

```powershell
vlm-distill create-manifest \
  --config configs/parsing_labeling.yaml \
```

Output:

```text
D:\TV_data\teacher_parsing\parsing_manifest.jsonl
```

Example:

```json
{
  "id":"parsing-000001",
  "image":"D:/TV_data/test_data/example.png",
  "task":"parsing",
  "query":"List all visible interactive UI elements on this screen."
}
```

---

## Step 2

Validate:

```powershell
vlm-distill validate-manifest \
  --config configs/parsing_labeling.yaml
```

---

## Step 3

Generate teacher labels:

```powershell
vlm-distill label \
  --config configs/parsing_labeling.yaml
```

Output:

```text
D:\TV_data\teacher_parsing\parsing_teacher_labels.jsonl
```

Expected teacher response:

```json
{
  "elements":[
    "YouTube",
    "Search"
  ]
}
```

Validate generated labels:

```powershell
  vlm-distill validate-teacher \
  --config configs/parsing_labeling.yaml
```

---

# Grounding Workflow

Grounding is automatically bootstrapped from Screen Parsing results.

No manual target label selection is required.

---

## Configuration

```yaml
data:
  output_dir: D:/TV_data/teacher_parsing

  manifest_path: D:/TV_data/teacher_parsing/grounding_manifest.jsonl

  distill_path: D:/TV_data/teacher_parsing/grounding_teacher_labels.jsonl

  eval_path: D:/TV_data/teacher_parsing/grounding_teacher_labels.jsonl
```

---

## Step 1

Generate grounding manifest:

```powershell
vlm-distill create-manifest \
  --config configs/grounding_test.yaml \
```

This automatically reads:

```text
parsing_teacher_labels.jsonl
```

and expands:

```json
{
  "label":"YouTube"
}
```

into:

```json
{
  "target_label":"YouTube",
  "target_type":"object"
}
```

---

## Step 2

Generate grounding teacher labels:

```powershell
vlm-distill label \
  --config configs/grounding_test.yaml
```

Expected output:

```json
{
  "label":"YouTube",
  "bbox":[100,200,300,400],
  "confidence":0.93
}
```

Output:

```text
D:\TV_data\teacher_parsing\grounding_teacher_labels.jsonl
```

Validate generated labels:

```powershell
  vlm-distill validate-teacher \
  --config configs/grounding_test.yaml
```

---

# Response Distillation Workflow

Use this workflow when you want to distill a larger teacher VLM into a smaller student with standard response distillation / SFT.

Full flow:

```text
images
  -> parsing_manifest.jsonl
  -> parsing_teacher_labels.jsonl
  -> response distillation training
  -> distilled 3B adapter
  -> evaluation report
```

## Step 1

Create the screen parsing manifest:

```powershell
vlm-distill create-manifest \
  --config configs/parsing_labeling.yaml \
```

Typical output:

```text
D:\TV_data\teacher_parsing\parsing_manifest.jsonl
```

## Step 2

Validate the manifest:

```powershell
vlm-distill validate-manifest \
  --config configs/parsing_labeling.yaml
```

## Step 3

Generate teacher labels:

```powershell
vlm-distill label \
  --config configs/parsing_labeling.yaml
```

Typical output:

```text
D:\TV_data\teacher_parsing\parsing_teacher_labels.jsonl
```

Example teacher row:

```json
{
  "id": "parsing-000001",
  "image": "D:/TV_data/test_data/example.png",
  "task": "parsing",
  "query": "List all visible interactive UI elements on this screen.",
  "teacher_answer": "{\"elements\": [\"YouTube\", \"Search\"]}",
  "teacher_confidence": 1.0
}
```

## Step 4

Prepare the response distillation config.

If you already have a teacher-labeled JSONL, you can start from this step directly.

Example config:

```yaml
data:
  manifest_path: D:/TV_data/teacher_parsing/parsing_manifest.jsonl
  distill_path: D:/TV_data/teacher_parsing/parsing_teacher_labels_1080p_8bit.jsonl
  eval_path: D:/TV_data/teacher_parsing/parsing_teacher_labels_1080p_8bit.jsonl
  image_root: .

student:
  model_name: Qwen/Qwen2.5-VL-3B-Instruct
  output_dir: outputs/parsing_response_1080p_8bit
  adapter_dir: outputs/parsing_response_1080p_8bit/adapter
  quantization: 4bit

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  mixed_precision: bf16
  max_length: 4096

distillation:
  method: response
  prompt_template: "query: {query}\nAnswer:"
```

Reference configs in this repo:

```text
configs/parsing_labeling.yaml
configs/parsing_response_distillation.yaml
```

These configs share the same option keys and derived profiles:

```text
quality
teacher_quantization
student_quantization
label_profile = {quality}_{teacher_quantization}
response_profile = {quality}_{teacher_quantization}_student_{student_quantization}
```

The response distillation config currently points to:

```text
manifest_path = D:/TV_data/teacher_parsing/parsing_manifest.jsonl
distill_path = D:/TV_data/teacher_parsing/parsing_teacher_labels_1080p_8bit.jsonl
student model = Qwen/Qwen2.5-VL-3B-Instruct
teacher labels = 1080p_8bit screen parsing outputs
```

## Step 5

Validate the response distillation inputs:

```powershell
vlm-distill validate-manifest --config configs/parsing_response_distillation.yaml

vlm-distill validate-teacher --config configs/parsing_response_distillation.yaml
```

## Step 6

Train the student:

```powershell
vlm-distill train --config configs/parsing_response_distillation.yaml
```

Typical artifact output:

```text
outputs/parsing_response_1080p_8bit/adapter
```

## Step 7

Evaluate the distilled student:

```powershell
vlm-distill evaluate \
  --config configs/parsing_response_distillation.yaml
```

Typical evaluation output:

```text
outputs/parsing_response_1080p_8bit/eval_report.json
```

What this does:

```text
teacher_answer JSONL
        ->
multimodal prompt + image
        ->
training target = teacher_answer
        ->
LoRA fine-tuning on the student VLM
```

Adapter / merge guidance:

* Training writes a LoRA adapter into `student.adapter_dir`; it does not overwrite the original student base model.
* While you are still comparing experiments, keep inference in adapter mode instead of merging.
* Only merge for deployment, and always write the merged weights into a new directory such as `outputs/.../merged-*`.
* Do not write merged weights back into the base model directory. Keeping the original student base untouched lets you merge other adapters later.

Export a standalone merged student model for inference:

```bash
vlm-distill merge-adapter --config configs/parsing_switch_kd.yaml
```

Use one main YAML for both training and inference. The recommended customer-facing flow is:

```bash
vlm-distill create-manifest --config configs/parsing_switch_kd.yaml --split training
vlm-distill label --config configs/parsing_switch_kd.yaml
vlm-distill switch-logits --config configs/parsing_switch_kd.yaml
vlm-distill train --config configs/parsing_switch_kd.yaml

vlm-distill merge-adapter --config configs/parsing_switch_kd.yaml

vlm-distill create-manifest --config configs/parsing_switch_kd.yaml --split inference
vlm-distill predict --config configs/parsing_switch_kd.yaml
```

Do not maintain separate merge and inference configs such as `parsing_switch_kd_merge.yaml` and `parsing_switch_kd_infer.yaml`.

`merge-adapter` always loads the base model from `student.model_name`, loads the PEFT adapter from `student.inference_adapter_path` or `student.adapter_dir`, and writes the merged model to `student.merged_model_path` or `student.output_dir/merged_model`.

`predict` chooses the model source automatically:

```text
if student.inference_model_path is set:
    use inference_model_path
    adapter behavior follows load_adapter / merge_adapter

elif student.merged_model_path exists:
    use merged_model_path
    disable adapter loading

else:
    use student.model_name
    adapter behavior follows load_adapter / merge_adapter
```

When `predict` uses `student.merged_model_path`, it disables adapter loading to avoid applying the adapter twice.

Recommended single-YAML student section:

```yaml
student:
  model_name: /home/phison/vlm_distill/models/Qwen3-VL-8B-Instruct
  output_dir: outputs/switch-kd/{task_name}_switch_kd_{response_profile}
  adapter_dir: outputs/switch-kd/{task_name}_switch_kd_{response_profile}/adapter
  merged_model_path: outputs/switch-kd/{task_name}_switch_kd_{response_profile}/merged_model
  inference_model_path:
  inference_adapter_path:
  load_adapter: true
  merge_adapter: false
```

Parallel teacher precompute helper:

```bash
bash scripts/run_parallel_switch_kd_precompute_4gpu.sh --clean-outputs teacher-precompute
```

This helper splits `outputs/switch-kd/parsing_manifest.jsonl` into four shard manifests, generates temporary configs in `configs/generated/`, launches four `teacher-precompute` workers with `CUDA_VISIBLE_DEVICES=0..3`, then merges shard teacher outputs back into `outputs/switch-kd/parsing_teacher_labels_480p_8bit.jsonl` after all workers succeed.

## Step 8

Batch test a merged student model.

Instead of using `infer_merged.py` one sample at a time, you can run the merged model on the whole manifest like `teacher-precompute`.

Example config:

```yaml
data:
  manifest_path: D:/TV_data/teacher_parsing/parsing_manifest.jsonl
  label_path: D:/TV_data/teacher_parsing/parsing_teacher_labels_480p_8bit.jsonl
  prediction_path: outputs/parsing_merged_predictions_480p_8bit.jsonl
  eval_path: D:/TV_data/teacher_parsing/parsing_teacher_labels_480p_8bit.jsonl
  image_root: .

teacher:
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  max_new_tokens: 128

student:
  model_name: Qwen/Qwen2.5-VL-3B-Instruct
  output_dir: outputs/parsing_response_480p_8bit_student_4bit
  adapter_dir: outputs/parsing_response_480p_8bit_student_4bit/adapter
  merged_model_path: outputs/student/merged_response_KD_480p_8bit
  inference_model_path:
  inference_adapter_path:
  load_adapter: true
  merge_adapter: false
  quantization: none

distillation:
  prompt_template: "query: {query}\nAnswer:"

evaluation:
  output_path: outputs/parsing_merged_eval_report_480p_8bit.json
```

Run batch prediction:

```powershell
vlm-distill predict \
  --config your_merged_eval_config.yaml
```

If `outputs/student/merged_response_KD_480p_8bit` already exists and `inference_model_path` is empty, `predict` will load that merged model automatically and log:

```text
prediction_model_source=merged_model_path
model_path=outputs/student/merged_response_KD_480p_8bit
adapter=disabled
```

Typical output:

```text
outputs/parsing_merged_predictions_480p_8bit.jsonl
```

Run batch evaluation:

```powershell
vlm-distill evaluate-predictions \
  --config your_merged_eval_config.yaml
```

Typical evaluation output:

```text
outputs/parsing_merged_eval_report_480p_8bit.json
```

What this does:

```text
manifest.jsonl
  ->
merged student model
  ->
prediction JSONL
  ->
evaluation against reference labels
```

Notes for Qwen2.5-VL:

* `teacher_answer` is the supervision target used during training.
* `training.image_resize` controls how the student-side training image is resized before encoding. In the generic response config, it follows `{quality}` by default.`r`n* 1080p images can expand into a large number of image tokens, so `max_length: 4096` is a safer starting point than `512`.
* If you compare multiple teacher label files first, keep the original teacher label JSONL for training; the compare JSONL is for analysis, not for student training.
* The response distillation training path in this repo expects the original teacher label JSONL, not the row-wise compare JSONL.
* If you want to measure the gap between the distilled 3B student and the 7B teacher, keep the teacher label JSONL as the evaluation reference and run `vlm-distill evaluate` after training.

---
# Switch-KD Workflow

Switch-KD training pipeline:

```bash
python -m vlm_distill.cli teacher-precompute --config configs/parsing_switch_kd.yaml
python -m vlm_distill.cli validate-teacher --config configs/parsing_switch_kd.yaml
python -m vlm_distill.cli switch-logits --config configs/parsing_switch_kd.yaml
python -m vlm_distill.cli train --config configs/parsing_switch_kd.yaml
```

For Switch-KD, use `teacher-precompute`, followed by `validate-teacher`,
`switch-logits`, and `train`. `distillation.teacher_logits: true` means teacher
logits are generated during `teacher-precompute` into `data.label_path`, the
canonical unified teacher output.

4-GPU precompute:

```bash
bash scripts/run_parallel_switch_kd_precompute_4gpu.sh --clean-outputs teacher-precompute
python -m vlm_distill.cli validate-teacher --config configs/parsing_switch_kd.yaml
bash scripts/run_parallel_switch_kd_precompute_4gpu.sh --clean-outputs switch-logits
```

Or:

```bash
bash scripts/run_parallel_switch_kd_precompute_4gpu.sh --clean-outputs all
```

Here `all` means `teacher-precompute` then `switch-logits`.

Training objective:

```text
L_total
=
LM Loss
+
DBiLD Loss
+
VSD Loss
```

---

# Typical Workflow

Teacher label generation only:

```powershell
python -m compileall src

vlm-distill create-manifest \
  --config configs/parsing_labeling.yaml \

vlm-distill validate-manifest \
  --config configs/parsing_labeling.yaml

vlm-distill label \
  --config configs/parsing_labeling.yaml
```

Screen Parsing + Grounding:

```powershell
vlm-distill create-manifest \
  --config configs/parsing_labeling.yaml \

vlm-distill label \
  --config configs/parsing_labeling.yaml

vlm-distill create-manifest \
  --config configs/grounding_test.yaml \

vlm-distill label \
  --config configs/grounding_test.yaml
```

---

# Notes

* Use `max_samples: 3~5` for initial debugging.
* Keep model weights outside the repository.
* Add model directories to `.gitignore`.
* Grounding currently depends on Screen Parsing output.
* Local Qwen2.5-VL-7B-Instruct is recommended as the first teacher model.
* RTX 4060Ti 16GB can run Qwen2.5-VL-7B-Instruct with 4bit quantization comfortably.
* Student/training sections remain in YAML because the project currently uses a unified configuration schema.
* Screen Parsing is currently the most mature workflow in this repository.
