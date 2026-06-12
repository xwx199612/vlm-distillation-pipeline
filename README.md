# VLM Distillation Pipeline

Vision-Language Model Distillation Pipeline for GUI Automation Testing.

This project is designed for Android TV, mobile devices, tablets, in-vehicle infotainment systems, and other GUI-driven products. The goal is to generate high-quality teacher labels from large VLMs and distill them into smaller deployable models.

---

# Current Roadmap

Current milestone:

```text
Screen Parsing → Auto Grounding Bootstrap Pipeline
```

Pipeline:

```text
Device Screenshot
        ↓
Screen Parsing Teacher
        ↓
UI Elements
        ↓
Grounding Teacher
        ↓
Bounding Boxes
        ↓
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
  --config configs/screen_parsing_test.yaml \
  --task screen_parsing
```

or

```powershell
vlm-distill create-manifest \
  --config configs/grounding_test.yaml \
  --task grounding
```

---

## Validate Dataset

```powershell
vlm-distill validate-data \
  --config configs/screen_parsing_test.yaml
```

---

## Generate Teacher Labels

```powershell
vlm-distill label \
  --config configs/screen_parsing_test.yaml
```

---

## Generate Teacher Logits

```powershell
vlm-distill teacher-logits \
  --config configs/switch_kd_4060ti.yaml
```

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

---

## Evaluate Student

```powershell
vlm-distill evaluate \
  --config configs/switch_kd_4060ti.yaml
```

---

# Screen Parsing Workflow

## Configuration

Example:

```yaml
data:
  image_dir: D:/TV_data/test_data

  output_dir: D:/TV_data/teacher_parsing

  manifest_path: D:/TV_data/teacher_parsing/screen_parsing_manifest.jsonl

  distill_path: D:/TV_data/teacher_parsing/screen_parsing_teacher_labels.jsonl

  eval_path: D:/TV_data/teacher_parsing/screen_parsing_teacher_labels.jsonl

  image_root: .

  max_samples: 5
```

---

## Step 1

Generate manifest:

```powershell
vlm-distill create-manifest \
  --config configs/screen_parsing_test.yaml \
  --task screen_parsing
```

Output:

```text
D:\TV_data\teacher_parsing\screen_parsing_manifest.jsonl
```

Example:

```json
{
  "id":"screen_parsing-000001",
  "image":"D:/TV_data/test_data/example.png",
  "task":"screen_parsing",
  "query":"List all visible UI icons, buttons, menu items, text labels, and actionable elements on this screen.",
  "metadata":{}
}
```

---

## Step 2

Validate:

```powershell
vlm-distill validate-data \
  --config configs/screen_parsing_test.yaml
```

---

## Step 3

Generate teacher labels:

```powershell
vlm-distill label \
  --config configs/screen_parsing_test.yaml
```

Output:

```text
D:\TV_data\teacher_parsing\screen_parsing_teacher_labels.jsonl
```

Expected teacher response:

```json
{
  "screen_type":"Android TV Home",

  "elements":[
    {
      "label":"YouTube",
      "type":"app_icon"
    },
    {
      "label":"Settings",
      "type":"icon"
    }
  ]
}
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
  --task grounding
```

This automatically reads:

```text
screen_parsing_teacher_labels.jsonl
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
  "target_type":"object",
  "source_screen_parsing_id":"screen-001",
  "metadata":{}
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

---

# Switch-KD Workflow

Switch-KD training pipeline:

```powershell
vlm-distill validate-data \
  --config configs/switch_kd_4060ti.yaml

vlm-distill label \
  --config configs/switch_kd_4060ti.yaml

vlm-distill teacher-logits \
  --config configs/switch_kd_4060ti.yaml

vlm-distill switch-logits \
  --config configs/switch_kd_4060ti.yaml

vlm-distill train \
  --config configs/switch_kd_4060ti.yaml

vlm-distill evaluate \
  --config configs/switch_kd_4060ti.yaml
```

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
  --config configs/screen_parsing_test.yaml \
  --task screen_parsing

vlm-distill validate-data \
  --config configs/screen_parsing_test.yaml

vlm-distill label \
  --config configs/screen_parsing_test.yaml
```

Screen Parsing + Grounding:

```powershell
vlm-distill create-manifest \
  --config configs/screen_parsing_test.yaml \
  --task screen_parsing

vlm-distill label \
  --config configs/screen_parsing_test.yaml

vlm-distill create-manifest \
  --config configs/grounding_test.yaml \
  --task grounding

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
