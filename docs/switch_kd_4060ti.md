# Switch-KD on RTX 4060 Ti 16GB

This note describes the practical Switch-KD setup used in this repo:

1. Visual-Switch Distillation (VSD)
2. Dynamic Bi-directional Logits Difference (DBiLD)
3. Standard autoregressive language modeling loss

## 16GB Baseline

This configuration is tuned for the 16GB version of RTX 4060 Ti. With 16GB, the pipeline can keep a 7B teacher and a 2B-3B student in play more comfortably, so the defaults are less conservative:

- Teacher logits / switch logits can use a larger top-k cache.
- Student can be a 2B-3B class VLM.
- 4-bit quantization + LoRA remains the recommended student setup.
- `batch_size: 1` is still the safest choice.
- `gradient_accumulation_steps: 8` is a better default than 16 for this card.
- `max_length: 512` is a reasonable starting point.
- DBiLD can use `top_k: 64`.

## Project Files

- `src/vlm_distill/stage_teacher_logits.py`
  - Generates cached teacher logits
  - Longer teacher outputs are practical on 16GB

- `src/vlm_distill/stage_visual_switch_logits.py`
  - Generates VSD switch logits
  - Student vision encoder -> student projector -> teacher LLM
  - Component paths can point to getters such as `get_input_embeddings`

- `src/vlm_distill/loss_switch_kd.py`
  - `SwitchKDLoss`
  - `dynamic_bidirectional_logits_difference`
  - causal LM loss

- `src/vlm_distill/stage_student_training.py`
  - Enables `SwitchKDTrainer` when `distillation.method: switch_kd`
  - Consumes cached `teacher_logits` and `switch_logits`

- `configs/switch_kd_4060ti.yaml`
  - 4060 Ti 16GB-oriented config

## Data Paths

The core distillation row can look like this:

```json
{
  "id": "sample-001",
  "image": "data/images/001.jpg",
  "query": "What is in the image?",
  "student_target": "a cup",
  "teacher_logits": [[[...]]],
  "switch_logits": [[[...]]]
}
```

- `teacher_logits` comes from normal teacher VLM forward.
- `switch_logits` comes from the VSD path where student visual outputs are routed into the teacher language pathway.

If these cached fields are missing, the trainer falls back to standard LM loss only.

## Teacher Logits Stage

Command:

```powershell
python -m vlm_distill.cli teacher-label --config configs\switch_kd_4060ti.yaml
```

This stage:

1. Loads the teacher model and processor.
2. Builds the multimodal prompt.
3. Runs prompt-only forward to record `teacher_logits_prompt_len`.
4. Runs full multimodal forward.
5. Stores `teacher_logits`, `teacher_logits_format`, `teacher_logits_prompt_len`, and `teacher_logits_vocab_size`.

With the 16GB profile, `teacher.max_new_tokens` is set higher so the teacher has more room to finish a useful answer.

## VSD Stage

Command:

```powershell
python -m vlm_distill.cli switch-label --config configs\switch_kd_4060ti.yaml
```

This stage:

1. Loads student vision and projector components.
2. Loads teacher text embedding and language model components.
3. Converts the image into student visual features.
4. Projects those features into the teacher embedding space.
5. Splices the visual embeddings into the teacher text sequence.
6. Runs the teacher LLM forward and caches `switch_logits`.

With the 16GB profile, the cached vocab can be larger, so more of the VSD distribution is retained.

## Training

Command:

```powershell
python -m vlm_distill.cli train --config configs\switch_kd_4060ti.yaml
```

Training combines:

- LM loss
- DBiLD on `teacher_logits`
- VSD loss on `switch_logits`

## Recommended Flow

```powershell
python -m vlm_distill.cli label --config configs\switch_kd_4060ti.yaml
python -m vlm_distill.cli teacher-label --config configs\switch_kd_4060ti.yaml
python -m vlm_distill.cli switch-label --config configs\switch_kd_4060ti.yaml
python -m vlm_distill.cli train --config configs\switch_kd_4060ti.yaml
```

## VSD Note

The VSD implementation is model-agnostic and resolves component paths from config. If auto-resolution fails, set:

```yaml
distillation:
  student_vision_path: model.vision_model
  student_projector_path: model.connector
  teacher_lm_path: language_model
  teacher_token_embedding_path: get_input_embeddings
  teacher_lm_head_path: lm_head
  visual_token_placeholder: "<|image_pad|>"
```

