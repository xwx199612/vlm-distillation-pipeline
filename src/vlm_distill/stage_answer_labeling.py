from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import time
from io import BytesIO
from dataclasses import asdict
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin

from .config_schema import PipelineConfig, format_prompt, resolve_label_path
from .data_manifest import VlmSample, read_jsonl
from .model_loading import apply_attn_implementation, resolve_model_path


class TeacherBackend(Protocol):
    def answer(self, sample: VlmSample) -> dict:
        ...


class MockTeacher:
    def answer(self, sample: VlmSample) -> dict:
        seed = f"{sample.id}:{sample.query}:{sample.answer or ''}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        confidence = 0.55 + (int(digest[:4], 16) / 0xFFFF) * 0.4

        if sample.answer:
            teacher_answer = sample.answer
        elif sample.task == "parsing":
            elements = sample.metadata.get("elements") if isinstance(sample.metadata, dict) else None
            teacher_answer = json.dumps(
                {
                    "elements": elements if isinstance(elements, list) else ["mock icon", "mock settings"],
                },
                ensure_ascii=False,
            )
        elif sample.task == "grounding":
            bbox = sample.metadata.get("bbox") if isinstance(sample.metadata, dict) else None
            teacher_answer = json.dumps(
                {
                    "label": sample.target_label or "target",
                    "bbox": bbox or [0, 0, 100, 100],
                },
                ensure_ascii=False,
            )
        else:
            teacher_answer = f"mock answer for {sample.task}"

        return {
            "teacher_answer": teacher_answer,
            "teacher_tokens": [],
            "teacher_confidence": round(confidence, 4),
            "teacher_rationale": "Mock backend used for pipeline validation.",
        }


class HuggingFaceTeacher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig
            try:
                from transformers import AutoModelForImageTextToText as AutoModelForVLM
            except ImportError:  # pragma: no cover - fallback for older transformers
                from transformers import AutoModelForVision2Seq as AutoModelForVLM
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers and bitsandbytes to use the Hugging Face teacher backend."
            ) from exc

        model_path = resolve_model_path(config.teacher.model_name)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        allowed_quantization = {"none", "4bit", "8bit"}

        if config.teacher.quantization not in allowed_quantization:
            raise ValueError(
                f"Unsupported teacher quantization: "
                f"{config.teacher.quantization}. "
                f"Allowed values: {sorted(allowed_quantization)}"
            )
        model_kwargs = {
            "device_map": config.teacher.device_map or "auto",
            "trust_remote_code": True,
        }
        apply_attn_implementation(model_kwargs, config.teacher.attn_implementation)

        if config.teacher.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif config.teacher.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            if config.teacher.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif config.teacher.torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif config.teacher.torch_dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForVLM.from_pretrained(
            model_path,
            **model_kwargs,
            local_files_only=True,
        )

    def answer(self, sample: VlmSample) -> dict:
        image_path = self.config.data.image_root / sample.image
        image = _load_teacher_image(image_path, self.config.teacher.image_resize)

        prompt = _format_prompt(self.config, sample)
        answer, generated_ids = self._generate(image=image, prompt=prompt, sample=sample)
        answer = _normalize_teacher_answer(sample, answer)
        chosen_ids = generated_ids

        if sample.task == "parsing" and _parsing_quality_score(answer) <= 2:
            retry_prompt = _build_parsing_retry_prompt(sample)
            retry_answer, retry_ids = self._generate(
                image=image,
                prompt=retry_prompt,
                sample=sample,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
            )
            retry_answer = _normalize_teacher_answer(sample, retry_answer)
            if _parsing_quality_score(retry_answer) >= _parsing_quality_score(answer):
                answer = retry_answer
                chosen_ids = retry_ids

        return {
            "teacher_answer": answer.strip(),
            "teacher_tokens": chosen_ids,
            "teacher_confidence": 1.0,
            "teacher_rationale": "Generated by Hugging Face teacher backend.",
        }

    def _generate(
        self,
        *,
        image,
        prompt: str,
        sample: VlmSample,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
    ) -> tuple[str, list[int]]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = {
            "do_sample": self.config.teacher.temperature > 0,
            "max_new_tokens": self.config.teacher.max_new_tokens,
        }
        if self.config.teacher.temperature > 0:
            generation_kwargs["temperature"] = self.config.teacher.temperature
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size is not None:
            generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        output_ids = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if sample.task == "parsing" and _looks_degenerate_screen_output(answer):
            return "", []
        token_ids = generated_ids[0].detach().cpu().tolist() if generated_ids.shape[0] > 0 else []
        return answer, token_ids

class OpenAICompatibleTeacher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._requests = None
        self._openai_client = None
        if not config.teacher.base_url:
            raise ValueError("teacher.base_url is required when backend='openai_compatible'.")

    def answer(self, sample: VlmSample) -> dict:
        image_path = self.config.data.image_root / sample.image
        image_data_url = _image_to_data_url(image_path, resize_mode=self.config.teacher.image_resize)
        prompt = _format_prompt(self.config, sample)

        payloads = [
            self._build_responses_payload(prompt, image_data_url),
            self._build_chat_payload(prompt, image_data_url),
        ]
        errors: list[str] = []
        for api_mode, payload in payloads:
            try:
                content = self._call_api(api_mode=api_mode, payload=payload)
                return {
                    "teacher_answer": content.strip(),
                    "teacher_tokens": [],
                    "teacher_confidence": 1.0,
                    "teacher_rationale": "Generated by OpenAI-compatible teacher backend.",
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{api_mode}: {exc}")
                continue

        raise RuntimeError(
            "OpenAI-compatible teacher backend failed for both responses and chat/completions. "
            + " | ".join(errors)
        )

    def _build_chat_payload(self, prompt: str, image_data_url: str) -> tuple[str, dict]:
        return (
            "chat/completions",
            {
                "model": self.config.teacher.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
                "temperature": self.config.teacher.temperature,
                "max_tokens": self.config.teacher.max_new_tokens,
            },
        )

    def _build_responses_payload(self, prompt: str, image_data_url: str) -> tuple[str, dict]:
        return (
            "responses",
            {
                "model": self.config.teacher.model_name,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_data_url},
                        ],
                    }
                ],
                "temperature": self.config.teacher.temperature,
                "max_output_tokens": self.config.teacher.max_new_tokens,
            },
        )

    def _call_api(self, *, api_mode: str, payload: dict) -> str:
        if _has_requests():
            return self._call_via_requests(api_mode=api_mode, payload=payload)
        if _has_openai():
            return self._call_via_openai(api_mode=api_mode, payload=payload)
        raise RuntimeError(
            "OpenAI-compatible backend requires either `requests` or `openai` to be installed."
        )

    def _call_via_requests(self, *, api_mode: str, payload: dict) -> str:
        requests = _import_requests()
        endpoint = f"/{api_mode}"
        response = requests.post(
            _join_url(self.config.teacher.base_url, endpoint),
            headers=_auth_headers(self.config.teacher.api_key),
            json=payload,
            timeout=self.config.teacher.request_timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
        return _extract_openai_compatible_text(data, api_mode=api_mode)

    def _call_via_openai(self, *, api_mode: str, payload: dict) -> str:
        client = _openai_client(self.config.teacher.base_url, self.config.teacher.api_key)
        if api_mode == "chat/completions":
            response = client.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                temperature=payload["temperature"],
                max_tokens=payload["max_tokens"],
                timeout=self.config.teacher.request_timeout,
            )
            return _extract_openai_sdk_text(response, api_mode=api_mode)
        if api_mode == "responses":
            response = client.responses.create(
                model=payload["model"],
                input=payload["input"],
                temperature=payload["temperature"],
                max_output_tokens=payload["max_output_tokens"],
                timeout=self.config.teacher.request_timeout,
            )
            return _extract_openai_sdk_text(response, api_mode=api_mode)
        raise ValueError(f"Unsupported api_mode: {api_mode}")


class OllamaTeacher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        if not config.teacher.ollama_host:
            raise ValueError("teacher.ollama_host must be set for backend='ollama'.")

    def answer(self, sample: VlmSample) -> dict:
        image_path = self.config.data.image_root / sample.image
        image_data = _image_to_base64(image_path, resize_mode=self.config.teacher.image_resize)
        prompt = _format_prompt(self.config, sample)
        payload = {
            "model": self.config.teacher.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.config.teacher.temperature,
                "num_predict": self.config.teacher.max_new_tokens,
            },
        }
        response = _import_requests().post(
            _join_url(self.config.teacher.ollama_host, "/api/chat"),
            json=payload,
            timeout=self.config.teacher.request_timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
        content = _extract_ollama_text(data)
        return {
            "teacher_answer": content.strip(),
            "teacher_tokens": [],
            "teacher_confidence": 1.0,
            "teacher_rationale": "Generated by Ollama teacher backend.",
        }


def build_teacher(config: PipelineConfig) -> TeacherBackend:
    if config.teacher.backend == "mock":
        return MockTeacher()
    if config.teacher.backend == "hf":
        return HuggingFaceTeacher(config)
    if config.teacher.backend == "openai_compatible":
        return OpenAICompatibleTeacher(config)
    if config.teacher.backend == "ollama":
        return OllamaTeacher(config)
    raise ValueError(f"Unknown teacher backend: {config.teacher.backend}")

def _format_prompt(config: PipelineConfig, sample: VlmSample) -> str:
    return format_prompt(
        config.distillation.prompt_template,
        query=sample.query,
        target_label=sample.target_label,
        target_type=sample.target_type,
        task=sample.task,
    )


def _target_from_existing_annotation(sample: VlmSample) -> str | None:
    elements = sample.metadata.get("elements") if isinstance(sample.metadata, dict) else None
    bbox = sample.metadata.get("bbox") if isinstance(sample.metadata, dict) else None

    if sample.task == "parsing" and elements:
        return json.dumps(
            elements if isinstance(elements, dict) else {"elements": elements},
            ensure_ascii=False,
        )

    if sample.task == "grounding" and bbox:
        return json.dumps(
            {
                "label": sample.target_label or "target object",
                "bbox": bbox,
            },
            ensure_ascii=False,
        )

    if sample.answer:
        return sample.answer

    return None


def create_distillation_dataset(config: PipelineConfig, samples: list[VlmSample]) -> Path:
    output_path = resolve_label_path(config.data)
    completed_ids = _load_completed_ids(output_path)
    total = len(samples)
    pending_samples = [sample for sample in samples if sample.id not in completed_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Labeling samples: total={total}, completed={len(completed_ids)}, "
        f"pending={len(pending_samples)}, output={output_path}"
    )

    if not pending_samples:
        print("No pending samples. Existing output is already complete for this manifest.")
        return output_path

    teacher = build_teacher(config)
    completed_now = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(pending_samples, start=1):
            started = time.perf_counter()
            row = _label_sample(config, teacher, sample)
            if row is None:
                print(
                    f"[label] skipped {index}/{len(pending_samples)} "
                    f"id={sample.id} reason=low_confidence"
                )
                continue

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed_now += 1
            elapsed = time.perf_counter() - started
            total_done = len(completed_ids) + completed_now
            print(
                f"[label] wrote {total_done}/{total} "
                f"id={sample.id} elapsed={elapsed:.2f}s"
            )

    return output_path


def _label_sample(
    config: PipelineConfig,
    teacher: TeacherBackend,
    sample: VlmSample,
) -> dict | None:
    existing_target = _target_from_existing_annotation(sample)

    if existing_target is not None:
        label = {
            "teacher_answer": existing_target,
            "teacher_tokens": [],
            "teacher_confidence": 1.0,
            "teacher_rationale": "Used existing manifest annotation.",
        }
    else:
        label = teacher.answer(sample)

    label["teacher_answer"] = _normalize_teacher_answer(sample, label["teacher_answer"])

    if label["teacher_confidence"] < config.distillation.min_teacher_confidence:
        return None

    return {
        **asdict(sample),
        **label,
    }


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed_ids: set[str] = set()
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is not None:
            completed_ids.add(str(sample_id))
    return completed_ids


def _load_teacher_image(image_path: Path, resize_mode: str):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    return _resize_teacher_image(image, resize_mode)


def _resize_teacher_image(image, resize_mode: str):
    mode = _normalize_image_resize_mode(resize_mode)
    if mode == "original":
        return image

    target_height = {
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
    }[mode]
    width, height = image.size
    if height <= target_height:
        return image

    target_width = round(width * target_height / height)
    return image.resize((target_width, target_height), _pil_lanczos())


def _resized_image_bytes(image_path: Path, resize_mode: str) -> bytes:
    image = _load_teacher_image(image_path, resize_mode)
    suffix = image_path.suffix.lower()
    image_format = "PNG" if suffix == ".png" else "JPEG"
    buffer = BytesIO()
    save_kwargs = {"format": image_format}
    if image_format == "JPEG":
        save_kwargs["quality"] = 95
    image.save(buffer, **save_kwargs)
    return buffer.getvalue()


def _normalize_image_resize_mode(resize_mode: str | None) -> str:
    mode = (resize_mode or "original").lower()
    aliases = {
        "none": "original",
        "no": "original",
        "off": "original",
        "native": "original",
        "original": "original",
        "1080": "1080p",
        "1080p": "1080p",
        "720": "720p",
        "720p": "720p",
        "480": "480p",
        "480p": "480p",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported teacher.image_resize={resize_mode!r}. "
            "Use one of: original, 480p, 720p, 1080p."
        )
    return aliases[mode]


def _pil_lanczos():
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _image_to_base64(image_path: Path, *, resize_mode: str = "original") -> str:
    if _normalize_image_resize_mode(resize_mode) == "original":
        return base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_bytes = _resized_image_bytes(image_path, resize_mode)
    return base64.b64encode(image_bytes).decode("ascii")


def _image_to_data_url(image_path: Path, *, resize_mode: str = "original") -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{_image_to_base64(image_path, resize_mode=resize_mode)}"


def _join_url(base_url: str, endpoint: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _has_requests() -> bool:
    return importlib.util.find_spec("requests") is not None


def _import_requests():
    if not _has_requests():
        raise RuntimeError(
            "The selected teacher backend requires the `requests` package. "
            "Install it in the environment or use the Hugging Face backend."
        )
    import requests

    return requests


def _has_openai() -> bool:
    return importlib.util.find_spec("openai") is not None


def _openai_client(base_url: str | None, api_key: str | None):
    if not _has_openai():
        raise RuntimeError(
            "OpenAI-compatible backend can use the `openai` package, but it is not installed."
        )
    from openai import OpenAI

    kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _extract_openai_compatible_text(data: dict, *, api_mode: str) -> str:
    if isinstance(data, dict):
        if "output_text" in data and data["output_text"]:
            return str(data["output_text"])
        if api_mode == "chat/completions":
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return str(content)
        if api_mode == "responses":
            if "output" in data:
                return _extract_openai_output_list(data["output"])
            if "content" in data:
                return _extract_openai_output_list(data["content"])
    raise RuntimeError(f"Could not parse OpenAI-compatible response payload: {data}")


def _extract_openai_sdk_text(response, *, api_mode: str) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)
    if api_mode == "chat/completions":
        choices = getattr(response, "choices", [])
        if choices:
            message = choices[0].message
            content = getattr(message, "content", "")
            if isinstance(content, list):
                return "".join(
                    getattr(part, "text", "") if not isinstance(part, str) else part
                    for part in content
                )
            return str(content)
    if api_mode == "responses":
        output = getattr(response, "output", None)
        if output is not None:
            return _extract_openai_output_list(output)
    raise RuntimeError(f"Could not parse OpenAI SDK response payload: {response}")


def _extract_openai_output_list(output) -> str:
    parts: list[str] = []
    for item in output:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(part))
            elif content is not None:
                parts.append(str(content))
        else:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(getattr(part, "text", part)))
            elif content is not None:
                parts.append(str(content))
    return "".join(parts)


def _extract_ollama_text(data: dict) -> str:
    message = data.get("message") or {}
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    if data.get("response") is not None:
        return str(data["response"])
    raise RuntimeError(f"Could not parse Ollama response payload: {data}")


_COMMON_TOP_TABS = {"search", "home", "shop", "discover", "apps"}
_SCREEN_SCHEMA_LABELS = {
    "",
    "true",
    "false",
    "null",
    "none",
    "id",
    "ref",
    "data",
    "version",
    "text",
    "label",
    "type",
    "icon",
    "button",
    "link",
    "tab",
    "tile",
    "toggle",
    "input",
    "menu item",
    "action",
    "elements",
    "element",
    "active navigation areas",
    "active navigation area",
    "selected active navigation area index",
    "top level tabs",
    "tab label",
    "is selected",
    "tab id",
    "is currently focused",
    "has focus indicators",
    "element type",
    "tab name",
    "is enabled",
    "tab title",
    "content items",
    "navigation elements",
    "ui elements",
    "additional ui features",
    "tile text",
    "tile icon",
    "tile plus button",
    "app tiles",
    "active",
    "selected",
    "status",
    "navigation",
    "content",
    "recommended",
    "navigation button",
    "content item",
    "text label",
}
_SCREEN_SCHEMA_SUBSTRINGS = (
    "schema",
    "json",
    "active navigation",
    "toplevel",
    "selectedtab",
    "tabfocused",
    "contentitems",
    "contentitemfocused",
    "uielements",
    "actionableelements",
    "focus indicators",
    "isactive",
    "isfocused",
    "tabindex",
    "active_top_tab",
    "focused_element",
    "focus_state",
)


def _normalize_teacher_answer(sample: VlmSample, teacher_answer: str) -> str:
    if sample.task != "parsing":
        return teacher_answer.strip()
    payload = _normalize_parsing_payload(teacher_answer)
    return json.dumps(payload, ensure_ascii=False)


def _normalize_parsing_payload(teacher_answer: str) -> dict[str, object]:
    payload = _empty_parsing_payload()
    parsed = _parse_json_object(teacher_answer)

    if isinstance(parsed, dict):
        raw_elements = parsed.get("elements")
        if raw_elements is None:
            raw_elements = parsed.get("selectable_elements")
        payload["elements"] = _normalize_screen_elements(raw_elements)

    if not payload["elements"]:
        payload["elements"] = _labels_to_screen_elements(_extract_candidate_labels(teacher_answer))

    return payload


def _empty_parsing_payload() -> dict[str, object]:
    return {
        "elements": [],
    }


def _parse_json_object(text: str) -> dict | None:
    candidate = text.strip()
    if not candidate:
        return None

    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _sanitize_screen_field(value: object) -> str:
    if value is None:
        return ""
    cleaned = _clean_screen_label(str(value))
    return "" if _is_screen_schema_label(cleaned) else cleaned


def _normalize_screen_elements(raw_elements: object) -> list[str]:
    if not isinstance(raw_elements, list):
        return []

    elements: list[str] = []
    seen: set[str] = set()
    for raw_element in raw_elements:
        label = ""
        if isinstance(raw_element, str):
            label = raw_element
        elif isinstance(raw_element, dict):
            label = str(
                raw_element.get("label")
                or raw_element.get("name")
                or raw_element.get("text")
                or raw_element.get("title")
                or ""
            )
        else:
            continue

        cleaned_label = _clean_screen_label(label)
        if not cleaned_label or _is_screen_schema_label(cleaned_label):
            continue

        lowered = cleaned_label.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        elements.append(cleaned_label)

    return elements


def _extract_candidate_labels(text: str) -> list[str]:
    candidates = re.findall(r'"([^"\n]{1,80})"', text)
    labels: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _clean_screen_label(candidate)
        if not cleaned or _is_screen_schema_label(cleaned):
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        labels.append(cleaned)
    return labels


def _labels_to_screen_elements(labels: list[str]) -> list[str]:
    elements: list[str] = []
    for label in labels:
        elements.append(label)
    return elements


def _clean_screen_label(value: str) -> str:
    cleaned = value.strip().strip(",.:;!?'\"`[]{}()")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _is_screen_schema_label(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _SCREEN_SCHEMA_LABELS:
        return True
    if re.fullmatch(r"\d+", lowered):
        return True
    if len(lowered) == 1 and lowered.isalpha():
        return True
    if re.fullmatch(r"[a-z]+(?:_[a-z]+)+", lowered):
        return True
    return any(token in lowered for token in _SCREEN_SCHEMA_SUBSTRINGS)


def _parsing_quality_score(answer: str) -> int:
    payload = _normalize_parsing_payload(answer)
    elements = payload.get("elements", [])
    return len(elements) * 2


def _looks_degenerate_screen_output(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    punctuation_ratio = sum(1 for char in stripped if not char.isalnum() and not char.isspace()) / max(len(stripped), 1)
    if punctuation_ratio > 0.6:
        return True
    if re.fullmatch(r"[!?.`~_\-=\s]+", stripped):
        return True
    return False


def _build_parsing_retry_prompt(sample: VlmSample) -> str:
    query = sample.query or "List all visible interactive UI elements on this screen."
    return (
        "You are parsing a GUI screenshot for a small student model distillation dataset.\n"
        f"Task: {query}\n"
        "Return raw JSON only.\n"
        "Use this exact schema:\n"
        '{"elements":[""]}\n'
        "Rules:\n"
        "- include only interactive items a user can focus, select, click, tap, or activate\n"
        "- exclude descriptive paragraphs, movie summaries, ads, and decorative text\n"
        "- do not copy instruction words, schema keys, booleans, or placeholder text into elements\n"
        "- each element must be a visible interactive label from the screenshot"
    )
