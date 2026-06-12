# VLM Distillation Pipeline

一套全新的 Vision-Language Model 蒸餾 pipeline，從資料清單、teacher pseudo-label 產生、student 監督式微調，到評估與匯出皆獨立運作。

## Pipeline Overview

```text
raw image/question data
        |
        v
manifest.jsonl  ->  teacher inference  ->  distill_dataset.jsonl
        |                                      |
        v                                      v
   validation                           student training
                                               |
                                               v
                                      eval + export adapter
```

## Features

- 支援 JSONL manifest 資料格式。
- teacher 可接 Hugging Face VLM，也可先用 mock backend 做管線測試。
- student 使用 LoRA/QLoRA 風格設定，預設走 Hugging Face `transformers` + `peft`。
- 蒸餾 loss 支援 hard target、soft target metadata 權重、answer/caption 混合任務。
- CLI 分成 `validate-data`、`label`、`train`、`evaluate` 四個階段。
- 設定集中於 YAML，方便替換 teacher/student/model/data path。

## Quick Start

```powershell
cd outputs\vlm-distillation-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

先用 mock teacher 跑通資料蒸餾：

```powershell
vlm-distill validate-data --config configs/mock.yaml
vlm-distill label --config configs/mock.yaml
vlm-distill train --config configs/mock.yaml
vlm-distill evaluate --config configs/mock.yaml
```

## Data Format

`manifest.jsonl` 每列一筆樣本：

```json
{"id":"sample-001","image":"examples/images/sample_001.jpg","question":"What object is on the table?","answer":"a cup","task":"vqa"}
```

蒸餾後的 `distill_dataset.jsonl`：

```json
{"id":"sample-001","image":"examples/images/sample_001.jpg","question":"What object is on the table?","student_target":"a cup","teacher_answer":"a cup","teacher_confidence":0.91,"task":"vqa"}
```

## Suggested Real Models

常見組合：

- Teacher: `Qwen/Qwen2.5-VL-7B-Instruct` 或更大的 instruct VLM。
- Student: `Qwen/Qwen2.5-VL-3B-Instruct`、`HuggingFaceTB/SmolVLM2-2.2B-Instruct`，或你自己的小型 VLM。

請依 GPU VRAM 調整 `configs/*.yaml` 的 batch size、LoRA rank、quantization 與 gradient checkpointing。

## Teacher Backends

Hugging Face online model id:

```yaml
teacher:
  backend: hf
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
```

Hugging Face local folder:

```yaml
teacher:
  backend: hf
  model_name: D:/models/Qwen2.5-VL-7B-Instruct
```

OpenAI-compatible local server:

```yaml
teacher:
  backend: openai_compatible
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  base_url: http://localhost:1234/v1
  api_key: local-test-key
```

Ollama local model:

```yaml
teacher:
  backend: ollama
  model_name: llava:7b
  ollama_host: http://localhost:11434
```

## Project Layout

```text
configs/
docs/
deploy/
  experimental/
  production/
examples/
src/vlm_distill/
tests/
```

Switch-KD 4060 Ti notes:

```text
docs/switch_kd_4060ti.md
configs/switch_kd_4060ti.yaml
src/vlm_distill/loss_switch_kd.py
```

## Deployment Code

實驗階段使用：

```powershell
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe deploy\experimental\infer_with_adapter.py `
  --base-model "Qwen/Qwen2.5-VL-3B-Instruct" `
  --adapter-path "outputs/student/adapter" `
  --image "examples/images/sample_001.jpg" `
  --question "What object is on the table?"
```

正式部署先合併 adapter：

```powershell
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe deploy\production\merge_adapter.py `
  --base-model "Qwen/Qwen2.5-VL-3B-Instruct" `
  --adapter-path "outputs/student/adapter" `
  --output-dir "outputs/student/merged"
```

再用 merged model 推論：

```powershell
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe deploy\production\infer_merged.py `
  --model-path "outputs/student/merged" `
  --image "examples/images/sample_001.jpg" `
  --question "What object is on the table?"
```

## Notes

這份專案骨架刻意不依賴既有資料夾。你只需要把資料放成 manifest JSONL，再把 config 中的路徑換成你的資料位置。

Current Distillation Workflow
Stage 1: Build Screen Parsing Dataset

Generate a manifest from a folder of screenshots:

vlm-distill create-manifest --task screen_parsing

Output:

data/screen_parsing_test.jsonl
Stage 2: Generate Teacher Labels

Run the teacher VLM to analyze screenshots and produce structured UI descriptions:

vlm-distill label --config configs/screen_parsing_test.yaml

Output:

outputs/screen_parsing_teacher_labels.jsonl
Stage 3: Build Grounding Dataset

Automatically convert detected UI elements into grounding tasks:

vlm-distill create-manifest --task grounding

Output:

data/grounding_test.jsonl
Stage 4: Generate Grounding Labels

Run the teacher VLM to predict object locations:

vlm-distill label --config configs/grounding_test.yaml

Output:

outputs/grounding_teacher_labels.jsonl
Stage 5: Train Student Model
vlm-distill train --config configs/grounding_test.yaml
Stage 6: Evaluate Student Model
vlm-distill evaluate --config configs/grounding_test.yaml
Quick Validation

Validate a manifest before labeling:

vlm-distill validate-data --config configs/screen_parsing_test.yaml

or

vlm-distill validate-data --config configs/grounding_test.yaml
Teacher Configuration

Example local Qwen2.5-VL-7B-Instruct teacher:

teacher:
  backend: hf
  model_name: D:/Models/Qwen2.5-VL-7B-Instruct
  device_map: auto
  torch_dtype: float16
  quantization: 4bit
  temperature: 0.0
  max_new_tokens: 256