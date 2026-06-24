from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urljoin

from .config_schema import PipelineConfig, format_prompt, resolve_label_path
from .data_manifest import VlmSample, read_jsonl, validate_manifest
from .device_utils import (
    batch_to_device,
    get_module_by_path,
    ensure_stage_uses_cuda,
    print_stage_model_debug,
    resolve_requested_device_map,
    select_model_input_device,
)
from .model_loading import apply_attn_implementation, resolve_model_path
from .stage_visual_switch_logits import _compact_adaptive_sequence_logits


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
        self._input_device = None
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
        requested_device_map = resolve_requested_device_map(
            config.teacher.device_map,
            quantization=config.teacher.quantization,
            role="teacher",
        )
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
            "device_map": requested_device_map,
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
        self._input_device = select_model_input_device(
            self.model,
            preferred_modules=(getattr(self.model, "visual", None),),
            label="Teacher",
        )
        print_stage_model_debug(
            stage_label="Teacher",
            model_path=model_path,
            quantization_mode=config.teacher.quantization,
            requested_device_map=requested_device_map,
            model=self.model,
            selected_input_device=self._input_device,
        )
        ensure_stage_uses_cuda(
            stage_label="Teacher",
            requested_device_map=requested_device_map,
            model=self.model,
            selected_input_device=self._input_device,
        )

    def answer(self, sample: VlmSample) -> dict:
        image_path = self.config.data.image_root / sample.image
        image = _load_teacher_image(image_path, self.config.teacher.image_resize)

        prompt = _format_prompt(self.config, sample)
        answer, _generated_ids = self._generate(image=image, prompt=prompt, sample=sample)
        answer = _normalize_teacher_answer(sample, answer)

        if sample.task == "parsing" and _parsing_quality_score(answer) <= 2:
            retry_prompt = _build_parsing_retry_prompt(sample)
            retry_answer, _retry_ids = self._generate(
                image=image,
                prompt=retry_prompt,
                sample=sample,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
            )
            retry_answer = _normalize_teacher_answer(sample, retry_answer)
            if _parsing_quality_score(retry_answer) >= _parsing_quality_score(answer):
                answer = retry_answer

        return {
            "teacher_answer": answer.strip(),
            "teacher_tokens": self.tokenize_teacher_answer(answer.strip()),
            "teacher_confidence": 1.0,
            "teacher_rationale": "Generated by Hugging Face teacher backend.",
        }

    def tokenize_teacher_answer(self, answer: str) -> list[int]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            encoded = self.processor(text=[answer], return_tensors=None)
            input_ids = encoded["input_ids"][0]
        else:
            input_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        return [int(token_id) for token_id in input_ids]

    def decode_teacher_tokens(self, token_ids: list[int]) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        decoder = tokenizer if tokenizer is not None else self.processor
        return decoder.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

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
        ).to(self._input_device)

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

    label["teacher_answer"] = _normalize_teacher_answer(sample, label["teacher_answer"]).strip()
    tokenizer = getattr(teacher, "tokenize_teacher_answer", None)
    if callable(tokenizer):
        label["teacher_tokens"] = tokenizer(label["teacher_answer"])

    decoder = getattr(teacher, "decode_teacher_tokens", None)
    _validate_generated_label(sample, label, decoder=decoder if callable(decoder) else None)

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
_ALLOWED_SCREEN_ELEMENT_TYPES = {
    "tab",
    "button",
    "app_icon",
    "app_tile",
    "menu_item",
    "tile",
    "toggle",
    "input",
    "icon",
    "link",
    "other",
    "unknown",
}


def _normalize_teacher_answer(sample: VlmSample, teacher_answer: str) -> str:
    if sample.task != "parsing":
        return teacher_answer.strip()
    payload = _normalize_parsing_payload(teacher_answer)
    return _compact_json(payload)


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


def _normalize_screen_elements(raw_elements: object) -> list[dict[str, object]]:
    if not isinstance(raw_elements, list):
        return []

    elements: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_element in raw_elements:
        label: object = ""
        element_type: object = "unknown"
        focused: object = False
        if isinstance(raw_element, str):
            label = raw_element
        elif isinstance(raw_element, dict):
            label = (
                raw_element.get("text")
                or raw_element.get("label")
                or raw_element.get("name")
                or raw_element.get("title")
                or ""
            )
            element_type = raw_element.get("type") or raw_element.get("role") or "unknown"
            focused = raw_element.get("focused", raw_element.get("focus", False))
        else:
            continue

        cleaned_label = _clean_screen_label(str(label))
        if not cleaned_label or _is_screen_schema_label(cleaned_label):
            continue

        lowered = cleaned_label.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized_type = _normalize_screen_element_type(element_type)
        if lowered in _COMMON_TOP_TABS and normalized_type in {"unknown", "other", "input"}:
            normalized_type = "tab"
        elements.append(
            {
                "text": cleaned_label,
                "type": normalized_type,
                "focused": _normalize_screen_element_focused(focused),
            }
        )

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


def _labels_to_screen_elements(labels: list[str]) -> list[dict[str, object]]:
    elements: list[dict[str, object]] = []
    for label in labels:
        elements.append({"text": label, "type": "unknown", "focused": False})
    return elements


def _normalize_screen_element_type(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"

    cleaned = _clean_screen_label(value)
    if not cleaned:
        return "unknown"

    snake = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    snake = re.sub(r"_+", "_", snake)
    if not snake:
        return "unknown"
    if snake in _ALLOWED_SCREEN_ELEMENT_TYPES:
        return snake
    if snake == "unknown":
        return "other"

    tokens = [token for token in snake.split("_") if token]
    token_set = set(tokens)

    if token_set & {"app", "application"}:
        return "app_icon"
    if token_set & {"tile", "card", "carousel", "recommend", "movie", "content", "poster", "banner"}:
        return "tile"
    if "menu" in token_set:
        return "menu_item"
    if token_set & {"nav", "navigation"}:
        return "tab"
    if token_set & {"search", "search_box", "searchbar", "search_bar", "input", "text_box", "textbox", "text"}:
        return "input"
    if token_set & {"toggle", "switch"}:
        return "toggle"
    if token_set & {"icon", "setting", "settings"}:
        return "icon"
    if "link" in token_set:
        return "link"
    if token_set & {"button", "btn"}:
        return "button"
    if token_set & {"text", "label", "image"}:
        return "other"

    if "unknown" in token_set:
        return "other"

    return "other"


def _normalize_screen_element_focused(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "focused", "selected", "active"}
    return bool(value)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _strip_special_tokens(text: str) -> str:
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    return text.strip()


def _canonicalize_teacher_answer(answer: str) -> str:
    parsed = _parse_json_object(_strip_special_tokens(answer))
    if parsed is None:
        raise ValueError("teacher_answer is not valid JSON")
    return _canonical_json(parsed)


def _validate_parsing_teacher_answer(answer: str) -> tuple[bool, str | None]:
    parsed = _parse_json_object(answer)
    if parsed is None:
        return False, "teacher_answer is not valid JSON"
    elements = parsed.get("elements")
    if not isinstance(elements, list):
        return False, "teacher_answer.elements is not a list"
    for index, element in enumerate(elements):
        if isinstance(element, str):
            return False, f"teacher_answer.elements[{index}] is a string-list item"
        if not isinstance(element, dict):
            return False, f"teacher_answer.elements[{index}] is not an object"
        missing = {"text", "type", "focused"} - set(element)
        if missing:
            return False, f"teacher_answer.elements[{index}] missing {sorted(missing)}"
        if not isinstance(element.get("text"), str):
            return False, f"teacher_answer.elements[{index}].text is not a string"
        if not isinstance(element.get("type"), str):
            return False, f"teacher_answer.elements[{index}].type is not a string"
        if not isinstance(element.get("focused"), bool):
            return False, f"teacher_answer.elements[{index}].focused is not a boolean"
    return True, None


def _validate_generated_label(sample: VlmSample, label: dict, *, decoder=None) -> None:
    if sample.task != "parsing":
        return

    valid, reason = _validate_parsing_teacher_answer(str(label.get("teacher_answer") or ""))
    if not valid:
        raise ValueError(f"{sample.id}: {reason}")

    teacher_tokens = label.get("teacher_tokens")
    if decoder is None or not teacher_tokens:
        return

    decoded = decoder([int(token_id) for token_id in teacher_tokens])
    answer_canonical = _canonicalize_teacher_answer(str(label["teacher_answer"]))
    decoded_canonical = _canonicalize_teacher_answer(decoded)
    if decoded_canonical != answer_canonical:
        raise ValueError(f"{sample.id}: decoded teacher_tokens do not match teacher_answer")


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
        '{"elements":[{"text":"","type":"unknown","focused":false}]}\n'
        "Rules:\n"
        "- include only interactive items a user can focus, select, click, tap, or activate\n"
        "- exclude descriptive paragraphs, movie summaries, ads, and decorative text\n"
        "- do not copy instruction words, schema keys, booleans, or placeholder text into elements\n"
        "- each element must be a visible interactive label from the screenshot"
    )



DistillationMode = Literal["response", "adaptive_topk", "switch_kd"]

INACTIVE_LOGIT = -1.0e4


@dataclass(frozen=True)
class CompletedLogitsRows:
    ids: set[str]
    valid_count: int
    invalid_count: int
    first_invalid_keys: list[str] | None


class TeacherLogitsGenerator:
    """
    Generate teacher distillation data in one pass.

    Supported modes:

    1. response
       - teacher.generate()
       - saves teacher_answer only
       - for normal response distillation / SFT

    2. adaptive_topk
       - teacher.generate(output_scores=True)
       - saves teacher_answer + adaptive top-k generation logits
       - for adaptive top-k logits distillation

    3. switch_kd
       - teacher.generate(output_scores=True)
       - saves teacher_answer + adaptive top-k logits + entropy + entropy weights
       - for Switch-KD style distillation data
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._input_device = None

    def load(self) -> None:
        if self.config.teacher.backend == "mock":
            return

        if self.config.teacher.backend != "hf":
            raise ValueError(
                "teacher precompute logits currently supports backend='hf' or backend='mock'. "
                f"Got backend={self.config.teacher.backend!r}."
            )

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:  # pragma: no cover - fallback for older transformers
            from transformers import AutoModelForVision2Seq as AutoModelForVLM

        model_path = resolve_model_path(self.config.teacher.model_name)
        requested_device_map = resolve_requested_device_map(
            self.config.teacher.device_map,
            quantization=self.config.teacher.quantization,
            role="teacher",
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": requested_device_map,
            "trust_remote_code": True,
        }
        apply_attn_implementation(model_kwargs, self.config.teacher.attn_implementation)

        if self.config.teacher.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.teacher.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            if self.config.teacher.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif self.config.teacher.torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif self.config.teacher.torch_dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForVLM.from_pretrained(
            model_path,
            **model_kwargs,
            local_files_only=True,
        ).eval()
        self._input_device = select_model_input_device(
            self._model,
            preferred_modules=(
                get_module_by_path(self._model, "model.visual"),
                get_module_by_path(self._model, "visual"),
                get_module_by_path(self._model, "model.language_model.embed_tokens"),
                get_module_by_path(self._model, "model.language_model"),
            ),
            label="Teacher",
        )
        print_stage_model_debug(
            stage_label="Teacher logits",
            model_path=model_path,
            quantization_mode=self.config.teacher.quantization,
            requested_device_map=requested_device_map,
            model=self._model,
            selected_input_device=self._input_device,
        )
        ensure_stage_uses_cuda(
            stage_label="Teacher logits",
            requested_device_map=requested_device_map,
            model=self._model,
            selected_input_device=self._input_device,
        )

    def generate_for_sample(
        self,
        sample: VlmSample,
        *,
        mode: DistillationMode,
    ) -> dict[str, Any]:
        if self.config.teacher.backend == "mock":
            return self._mock_generate_for_sample(sample, mode=mode)

        if self._model is None or self._processor is None:
            self.load()

        import torch

        image_path = self.config.data.image_root / sample.image
        image = _load_teacher_image(
            image_path,
            self.config.teacher.image_resize,
        )
        prompt = _format_prompt(self.config, sample)

        with torch.no_grad():
            inputs = self._build_multimodal_inputs(image, prompt)
            prompt_len = int(inputs["input_ids"].shape[1])
            inputs = batch_to_device(inputs, self._input_device)

            include_scores = mode in {"adaptive_topk", "switch_kd"}
            generation = self._generate(inputs, include_scores=include_scores)

            generated_ids, scores = _extract_generated_ids_and_scores(
                generation,
                prompt_len=prompt_len,
                include_scores=include_scores,
            )

            answer = self._decode(generated_ids).strip()
            normalized_answer = _normalize_teacher_answer(sample, answer).strip()
            teacher_tokens = self.tokenize_teacher_answer(normalized_answer)

            result: dict[str, Any] = {
                "teacher_answer": answer,
                "teacher_confidence": 1.0,
                "teacher_rationale": f"Generated by Hugging Face teacher in {mode} mode.",
                "distillation_mode": mode,
                "teacher_generated_ids": generated_ids.detach().cpu().tolist(),
                "teacher_tokens": teacher_tokens,
            }
            result["teacher_answer"] = normalized_answer

            if mode == "response":
                return result

            if not scores:
                raise ValueError(
                    "Teacher generation did not return scores. "
                    "Cannot build logits distillation dataset."
                )

            raw_tokens = _flatten_generated_ids(generated_ids.detach().cpu().tolist())
            try:
                raw_matches = (
                    raw_tokens == teacher_tokens
                    and _canonicalize_teacher_answer(answer) == _canonicalize_teacher_answer(normalized_answer)
                )
            except ValueError:
                raw_matches = False
            if raw_matches and len(scores) == len(teacher_tokens):
                logits_payload = self._build_generation_logits_payload(
                    scores=scores,
                    mode=mode,
                    prompt_len=0,
                    source="generation_scores",
                )
            else:
                logits_payload = compute_teacher_forced_answer_logits(
                    image=image,
                    prompt=prompt,
                    teacher_answer=normalized_answer,
                    teacher_tokens=teacher_tokens,
                    model=self._model,
                    processor=self._processor,
                    config=self.config,
                )
            result.update(logits_payload)
            return result

    def tokenize_teacher_answer(self, answer: str) -> list[int]:
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            encoded = self._processor(text=[answer], return_tensors=None)
            input_ids = encoded["input_ids"][0]
        else:
            input_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        return [int(token_id) for token_id in input_ids]

    def _generate(self, inputs: dict[str, Any], *, include_scores: bool):
        temperature = float(self.config.teacher.temperature)
        do_sample = temperature > 0

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.teacher.max_new_tokens),
            "do_sample": do_sample,
        }

        if do_sample:
            generate_kwargs["temperature"] = temperature

        if include_scores:
            generate_kwargs["output_scores"] = True
            generate_kwargs["return_dict_in_generate"] = True

        return self._model.generate(
            **inputs,
            **generate_kwargs,
        )

    def _decode(self, generated_ids):
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _build_multimodal_inputs(self, image, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

    def _build_generation_logits_payload(
        self,
        *,
        scores: list[Any],
        mode: DistillationMode,
        prompt_len: int,
        source: str = "generation_scores",
    ) -> dict[str, Any]:
        field = self.config.distillation.teacher_logits_field

        compact = _compact_adaptive_generation_scores(
            scores=scores,
            mode=mode,
            base_k=int(self.config.distillation.dbild_top_k),
            max_cached_logits_vocab=self.config.distillation.max_cached_logits_vocab,
            temperature=float(self.config.distillation.kd_temperature),
        )

        return {
            field: compact,
            f"{field}_format": mode,
            f"{field}_prompt_len": prompt_len,
            f"{field}_vocab_size": compact["vocab_size"],
            f"{field}_aligned_to_answer": True,
            f"{field}_source": source,
            f"{field}_temperature": float(self.config.distillation.kd_temperature),
        }

    def _mock_generate_for_sample(
        self,
        sample: VlmSample,
        *,
        mode: DistillationMode,
    ) -> dict[str, Any]:
        answer = _mock_answer(sample)
        answer = _normalize_teacher_answer(sample, answer).strip()
        teacher_tokens = [ord(char) for char in answer]

        result: dict[str, Any] = {
            **asdict(sample),
            "teacher_answer": answer,
            "teacher_confidence": 1.0,
            "teacher_rationale": f"Mock teacher used in {mode} mode.",
            "distillation_mode": mode,
            "teacher_generated_ids": [[1, 2, 3]],
            "teacher_tokens": teacher_tokens,
        }

        if mode == "response":
            return result

        field = (
            self.config.distillation.switch_logits_field
            if mode == "switch_kd"
            else self.config.distillation.teacher_logits_field
        )

        steps = len(teacher_tokens)
        vocab_size = 16
        base_k = int(self.config.distillation.dbild_top_k)
        max_k = min(vocab_size, max(2, min(base_k, 8)))

        indices = []
        values = []
        entropy = []
        token_k = []
        entropy_weight = []

        for _ in range(steps):
            step_indices = list(range(max_k))
            step_values = [5.0 - rank for rank in range(max_k)]
            step_entropy = 1.0
            step_k = min(max_k, max(2, base_k))

            indices.append(step_indices)
            values.append(step_values)
            entropy.append(step_entropy)
            token_k.append(step_k)
            entropy_weight.append(_entropy_to_weight(step_entropy))

        compact: dict[str, Any] = {
            "indices": [indices],
            "values": [values],
            "shape": [1, steps, vocab_size],
            "vocab_size": vocab_size,
            "token_k": [token_k],
            "entropy": [entropy],
            "adaptive": True,
        }

        if mode == "switch_kd":
            compact["entropy_weight"] = [entropy_weight]
            compact["switch_kd"] = True

        result.update(
            {
                field: compact,
                f"{field}_format": mode,
                f"{field}_prompt_len": 0,
                f"{field}_vocab_size": vocab_size,
                f"{field}_aligned_to_answer": True,
                f"{field}_source": "teacher_forcing_forward",
                f"{field}_temperature": float(self.config.distillation.kd_temperature),
            }
        )
        return result


def create_teacher_precompute_dataset(config: PipelineConfig, samples: list[VlmSample] | None = None) -> Path:
    require_logits = bool(getattr(config.distillation, "teacher_logits", True))
    samples = samples or validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )
    output_path = resolve_label_path(config.data)
    completed = _load_completed_teacher_rows(output_path, config=config, require_logits=require_logits)
    if completed.invalid_count:
        _rewrite_valid_teacher_rows(output_path, config=config, require_logits=require_logits)
    completed_ids = completed.ids
    pending_samples = [sample for sample in samples if sample.id not in completed_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Unified teacher precompute:")
    print(f"  distillation.method: {_resolve_distillation_mode(config)}")
    print(f"  distillation.teacher_logits: {require_logits}")
    print(f"  label_path: {output_path}")
    print("  teacher_logits_path: deprecated; canonical output is label_path")
    print("  unified_output: single_path")
    print(f"  total samples: {len(samples)}")
    print(f"  valid completed rows: {completed.valid_count}")
    print(f"  invalid stale rows: {completed.invalid_count}")
    print(f"  pending rows: {len(pending_samples)}")
    if completed.first_invalid_keys:
        print(f"  first invalid row id/reason: {completed.first_invalid_keys}")

    if not pending_samples:
        return output_path

    if config.teacher.backend == "hf" and require_logits:
        generator: Any = TeacherLogitsGenerator(config)
        mode = _resolve_teacher_logits_mode(config)
    else:
        generator = build_teacher(config)
        mode = "response"

    completed_now = 0
    with output_path.open("a", encoding="utf-8") as label_handle:
        for sample in pending_samples:
            started = time.perf_counter()
            if isinstance(generator, TeacherLogitsGenerator):
                generated = generator.generate_for_sample(sample, mode=mode)
            else:
                generated = _generate_label_with_optional_mock_logits(
                    config,
                    generator,
                    sample,
                    include_logits=require_logits,
                )
            row = {**asdict(sample), **generated}
            if require_logits:
                _assert_teacher_logits_answer_length(row, config.distillation.teacher_logits_field)
            encoded = json.dumps(row, ensure_ascii=False) + "\n"
            label_handle.write(encoded)
            label_handle.flush()
            completed_now += 1
            elapsed = time.perf_counter() - started
            print(
                "[teacher-precompute] "
                f"total={len(samples)} completed={len(completed_ids) + completed_now} "
                f"pending={len(pending_samples) - completed_now} id={sample.id} "
                f"elapsed_seconds_per_sample={elapsed:.2f}"
            )
    return output_path


def create_distillation_dataset(config: PipelineConfig, samples: list[VlmSample]) -> Path:
    return create_teacher_precompute_dataset(config, samples)


def _generate_label_with_optional_mock_logits(
    config: PipelineConfig,
    teacher: Any,
    sample: VlmSample,
    *,
    include_logits: bool,
) -> dict[str, Any]:
    label = teacher.answer(sample)
    answer = _normalize_teacher_answer(sample, str(label["teacher_answer"])).strip()
    tokenizer = getattr(teacher, "tokenize_teacher_answer", None)
    if callable(tokenizer):
        tokens = [int(token_id) for token_id in tokenizer(answer)]
    else:
        tokens = [ord(char) for char in answer]
    row = {
        "teacher_answer": answer,
        "teacher_tokens": tokens,
        "teacher_confidence": float(label.get("teacher_confidence", 1.0)),
        "teacher_rationale": label.get("teacher_rationale", "Generated by teacher backend."),
        "distillation_mode": _resolve_teacher_logits_mode(config),
    }
    if include_logits:
        row.update(_mock_answer_only_logits_payload(config, len(tokens), source="teacher_forcing_forward"))
    return row


def _mock_answer_only_logits_payload(config: PipelineConfig, answer_len: int, *, source: str) -> dict[str, Any]:
    field = config.distillation.teacher_logits_field
    vocab_size = 16
    max_k = min(vocab_size, max(1, min(int(config.distillation.dbild_top_k), 8)))
    indices = []
    values = []
    token_k = []
    entropy = []
    for step_index in range(answer_len):
        step_indices = [(step_index + offset) % vocab_size for offset in range(max_k)]
        indices.append(step_indices)
        values.append([5.0 - rank for rank in range(max_k)])
        token_k.append(max_k)
        entropy.append(1.0)
    compact = {
        "indices": [indices],
        "values": [values],
        "shape": [1, answer_len, vocab_size],
        "vocab_size": vocab_size,
        "token_k": [token_k],
        "entropy": [entropy],
        "adaptive": True,
    }
    return {
        field: compact,
        f"{field}_format": _resolve_teacher_logits_mode(config),
        f"{field}_prompt_len": 0,
        f"{field}_vocab_size": vocab_size,
        f"{field}_aligned_to_answer": True,
        f"{field}_source": source,
        f"{field}_temperature": float(config.distillation.kd_temperature),
    }


def compute_teacher_forced_answer_logits(
    *,
    image,
    prompt: str,
    teacher_answer: str,
    teacher_tokens: list[int],
    model,
    processor,
    config: PipelineConfig,
) -> dict[str, Any]:
    import torch

    prompt_inputs = _build_multimodal_inputs_for_processor(processor, image, prompt)
    full_inputs = _build_multimodal_inputs_for_processor(
        processor,
        image,
        _join_prompt_and_answer(prompt, teacher_answer),
    )
    input_device = select_model_input_device(model, label="Teacher forcing")
    prompt_inputs = batch_to_device(prompt_inputs, input_device)
    full_inputs = batch_to_device(full_inputs, input_device)
    prefix_len = int(prompt_inputs["input_ids"].shape[1])
    answer_len = len(teacher_tokens)
    with torch.no_grad():
        outputs = model(**full_inputs)
    full_logits = outputs.logits
    answer_logits = full_logits[:, prefix_len - 1 : prefix_len - 1 + answer_len, :]
    if int(answer_logits.shape[1]) != answer_len:
        raise ValueError(
            "Teacher-forced logits length mismatch: "
            f"answer_logits_len={int(answer_logits.shape[1])}, teacher_tokens_len={answer_len}"
        )
    compact = _compact_adaptive_sequence_logits(
        answer_logits,
        base_k=int(config.distillation.dbild_top_k),
        max_cached_logits_vocab=config.distillation.max_cached_logits_vocab,
        temperature=float(config.distillation.kd_temperature),
    )
    field = config.distillation.teacher_logits_field
    if int(compact["shape"][1]) != answer_len:
        raise ValueError("Compacted teacher logits are not aligned to teacher_tokens.")
    return {
        field: compact,
        f"{field}_format": _resolve_teacher_logits_mode(config),
        f"{field}_prompt_len": 0,
        f"{field}_vocab_size": int(compact["vocab_size"]),
        f"{field}_aligned_to_answer": True,
        f"{field}_source": "teacher_forcing_forward",
        f"{field}_temperature": float(config.distillation.kd_temperature),
    }


def _build_multimodal_inputs_for_processor(processor, image, text: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    templated = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(text=[templated], images=[image], return_tensors="pt")


def _join_prompt_and_answer(prompt: str, answer: str) -> str:
    separator = "" if prompt.endswith((" ", "\n")) else " "
    return f"{prompt}{separator}{answer}".strip()


def _assert_teacher_logits_answer_length(row: dict[str, Any], field_name: str) -> None:
    from .teacher_validation import validate_teacher_row
    valid, reason = validate_teacher_row(row, require_teacher_logits=True, logits_field=field_name)
    if not valid:
        raise ValueError(f"Unified teacher row failed validation id={row.get('id')}: {reason}")


def _load_completed_teacher_rows(
    path: Path,
    *,
    config: PipelineConfig,
    require_logits: bool,
) -> CompletedLogitsRows:
    from .teacher_validation import build_teacher_token_decoder, validate_teacher_row
    if not path.exists():
        return CompletedLogitsRows(ids=set(), valid_count=0, invalid_count=0, first_invalid_keys=None)
    completed_ids: set[str] = set()
    valid_count = 0
    invalid_count = 0
    first_invalid: list[str] | None = None
    decoder = build_teacher_token_decoder(config)
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is None:
            continue
        valid, reason = validate_teacher_row(
            row,
            require_teacher_logits=require_logits,
            decode_tokens=decoder,
        )
        if valid:
            completed_ids.add(str(sample_id))
            valid_count += 1
        else:
            invalid_count += 1
            if first_invalid is None:
                first_invalid = [str(sample_id), str(reason)]
    return CompletedLogitsRows(
        ids=completed_ids,
        valid_count=valid_count,
        invalid_count=invalid_count,
        first_invalid_keys=first_invalid,
    )


def _rewrite_valid_teacher_rows(path: Path, *, config: PipelineConfig, require_logits: bool) -> None:
    from .teacher_validation import build_teacher_token_decoder, validate_teacher_row
    decoder = build_teacher_token_decoder(config)
    valid_rows = [
        row for row in read_jsonl(path)
        if validate_teacher_row(
            row,
            require_teacher_logits=require_logits,
            decode_tokens=decoder,
        )[0]
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[teacher-precompute] pruned invalid existing rows from {path}; remaining_valid_rows={len(valid_rows)}")


def _load_completed_ids(
    path: Path,
    *,
    field_name: str | None = None,
    require_logits: bool = False,
) -> CompletedLogitsRows:
    if not path.exists():
        return CompletedLogitsRows(ids=set(), valid_count=0, invalid_count=0, first_invalid_keys=None)

    completed_ids: set[str] = set()
    valid_count = 0
    invalid_count = 0
    first_invalid_keys: list[str] | None = None
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is None:
            continue
        if require_logits and field_name is not None and not _is_valid_logits_row(row, field_name):
            invalid_count += 1
            if first_invalid_keys is None:
                first_invalid_keys = sorted(str(key) for key in row.keys())
            continue
        completed_ids.add(str(sample_id))
        valid_count += 1
    return CompletedLogitsRows(
        ids=completed_ids,
        valid_count=valid_count,
        invalid_count=invalid_count,
        first_invalid_keys=first_invalid_keys,
    )


def _rewrite_valid_completed_rows(path: Path, *, field_name: str) -> None:
    valid_rows = [row for row in read_jsonl(path) if _is_valid_logits_row(row, field_name)]
    with path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[teacher-precompute] pruned invalid existing logits rows from {path}; "
        f"remaining_valid_rows={len(valid_rows)}"
    )


def _is_valid_logits_row(row: dict[str, Any], field_name: str) -> bool:
    payload = row.get(field_name)
    if not isinstance(payload, dict):
        return False
    if not all(key in payload for key in ("indices", "values", "vocab_size")):
        return False
    indices_shape = _nested_shape(payload.get("indices"))
    values_shape = _nested_shape(payload.get("values"))
    if not indices_shape or not values_shape:
        return False
    return indices_shape == values_shape


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        return ()
    first_shape = _nested_shape(value[0])
    for item in value[1:]:
        if _nested_shape(item) != first_shape:
            return ()
    return (len(value), *first_shape)


def _validate_first_teacher_logits_row(
    row: dict[str, Any],
    *,
    field_name: str,
    method: DistillationMode,
    mode: DistillationMode,
    include_scores: bool,
    output_path: Path,
) -> None:
    if _is_valid_logits_row(row, field_name):
        return
    raise ValueError(
        "Teacher logits output row is missing a valid logits payload. "
        f"method={method}, resolved_mode={mode}, include_scores={include_scores}, "
        f"output_path={output_path}, first_row_keys={sorted(row.keys())}"
    )


def _flatten_generated_ids(generated_ids: Any) -> list[int]:
    if isinstance(generated_ids, list) and len(generated_ids) == 1 and isinstance(generated_ids[0], list):
        return [int(value) for value in generated_ids[0]]
    if isinstance(generated_ids, list):
        return [int(value) for value in generated_ids]
    return []


def _resolve_teacher_logits_mode(config: PipelineConfig) -> DistillationMode:
    mode = _resolve_distillation_mode(config)
    if mode == "switch_kd":
        return "adaptive_topk"
    return mode


def _resolve_distillation_mode(config: PipelineConfig) -> DistillationMode:
    raw_mode = str(
        getattr(config.distillation, "mode", None)
        or getattr(config.distillation, "method", "response")
        or "response"
    ).strip().lower()

    aliases = {
        "sft": "response",
        "response": "response",
        "response_distillation": "response",
        "response-distillation": "response",
        "topk": "adaptive_topk",
        "topk_logits": "adaptive_topk",
        "top-k": "adaptive_topk",
        "top-k-logits": "adaptive_topk",
        "adaptive_topk": "adaptive_topk",
        "adaptive-topk": "adaptive_topk",
        "adaptive_topk_logits": "adaptive_topk",
        "adaptive-topk-logits": "adaptive_topk",
        "dbild": "adaptive_topk",
        "switch": "switch_kd",
        "switch_kd": "switch_kd",
        "switch-kd": "switch_kd",
    }

    if raw_mode not in aliases:
        raise ValueError(
            f"Unsupported distillation method: {raw_mode!r}. "
            "Expected one of: response, adaptive_topk, switch_kd."
        )

    return aliases[raw_mode]  # type: ignore[return-value]


def _format_prompt(config: PipelineConfig, sample: VlmSample) -> str:
    template = config.distillation.prompt_template

    try:
        return template.format(
            query=sample.query or "",
            question=sample.query or "",
            target_label=sample.target_label or "",
            target_type=sample.target_type or "",
            task=sample.task,
        )
    except KeyError as exc:
        raise KeyError(
            f"Prompt template references unsupported placeholder: {exc}. "
            "Supported placeholders are: query, question, target_label, target_type, task."
        ) from exc


def _compact_adaptive_generation_scores(
    *,
    scores: list[Any],
    mode: DistillationMode,
    base_k: int,
    max_cached_logits_vocab: int | None,
    temperature: float,
) -> dict[str, Any]:
    """
    Convert generation scores into a compact adaptive top-k cache.

    Input:
      scores: list of tensors, each shaped [batch, vocab]

    Output:
      {
        "indices": [batch, generated_steps, max_k],
        "values": [batch, generated_steps, max_k],
        "shape": [batch, generated_steps, vocab],
        "vocab_size": vocab,
        "token_k": [batch, generated_steps],
        "entropy": [batch, generated_steps],
        "entropy_weight": [batch, generated_steps],  # switch_kd only
      }

    Notes:
      - We keep a rectangular [B, T, max_k] structure so existing cache
        materialization code can still scatter indices/values.
      - For each token, only the first token_k entries are active.
      - Inactive entries are filled with INACTIVE_LOGIT.
    """

    import torch

    if not scores:
        raise ValueError("Cannot compact empty generation scores.")

    first = scores[0].detach().float().cpu()
    if first.ndim != 2:
        raise ValueError(
            f"Expected each generation score to have shape [batch, vocab], got {tuple(first.shape)}"
        )

    batch_size, vocab_size = first.shape

    base_k = max(1, int(base_k))
    low_k = max(1, base_k // 4)
    mid_k = base_k
    high_k = base_k * 2

    if max_cached_logits_vocab is not None:
        high_k = min(high_k, int(max_cached_logits_vocab))

    max_k = min(vocab_size, max(low_k, mid_k, high_k))

    low_entropy_threshold = 1.0
    high_entropy_threshold = 2.5

    step_indices: list[Any] = []
    step_values: list[Any] = []
    step_entropy: list[Any] = []
    step_token_k: list[Any] = []
    step_entropy_weight: list[Any] = []

    safe_temperature = max(float(temperature), 1e-6)

    for score in scores:
        logits = score.detach().float().cpu()
        if logits.ndim != 2:
            raise ValueError(
                f"Expected each generation score to have shape [batch, vocab], got {tuple(logits.shape)}"
            )

        if logits.shape[0] != batch_size or logits.shape[1] != vocab_size:
            raise ValueError(
                "All generation scores must share the same [batch, vocab] shape. "
                f"Expected {(batch_size, vocab_size)}, got {tuple(logits.shape)}"
            )

        scaled_logits = logits / safe_temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

        top_values, top_indices = torch.topk(
            logits,
            k=max_k,
            dim=-1,
        )

        token_k = torch.empty((batch_size,), dtype=torch.long)
        entropy_weight = torch.empty((batch_size,), dtype=torch.float32)

        for batch_index in range(batch_size):
            entropy_value = float(entropy[batch_index].item())
            active_k = _adaptive_k(
                entropy_value,
                low_entropy_threshold=low_entropy_threshold,
                high_entropy_threshold=high_entropy_threshold,
                low_k=low_k,
                mid_k=mid_k,
                high_k=high_k,
                max_k=max_k,
            )

            token_k[batch_index] = active_k
            entropy_weight[batch_index] = _entropy_to_weight(entropy_value)

            if active_k < max_k:
                top_values[batch_index, active_k:] = INACTIVE_LOGIT

        step_indices.append(top_indices)
        step_values.append(top_values)
        step_entropy.append(entropy)
        step_token_k.append(token_k)
        step_entropy_weight.append(entropy_weight)

    indices_tensor = torch.stack(step_indices, dim=1)
    values_tensor = torch.stack(step_values, dim=1)
    entropy_tensor = torch.stack(step_entropy, dim=1)
    token_k_tensor = torch.stack(step_token_k, dim=1)

    compact: dict[str, Any] = {
        "indices": indices_tensor.tolist(),
        "values": values_tensor.tolist(),
        "shape": [batch_size, len(scores), vocab_size],
        "vocab_size": int(vocab_size),
        "adaptive": True,
        "token_k": token_k_tensor.tolist(),
        "entropy": entropy_tensor.tolist(),
        "k_policy": {
            "low_entropy_threshold": low_entropy_threshold,
            "high_entropy_threshold": high_entropy_threshold,
            "low_k": int(low_k),
            "mid_k": int(mid_k),
            "high_k": int(high_k),
            "max_k": int(max_k),
        },
    }

    if mode == "switch_kd":
        entropy_weight_tensor = torch.stack(step_entropy_weight, dim=1)
        compact["entropy_weight"] = entropy_weight_tensor.tolist()
        compact["switch_kd"] = True

    return compact


def _adaptive_k(
    entropy: float,
    *,
    low_entropy_threshold: float,
    high_entropy_threshold: float,
    low_k: int,
    mid_k: int,
    high_k: int,
    max_k: int,
) -> int:
    if entropy < low_entropy_threshold:
        return min(max_k, low_k)

    if entropy < high_entropy_threshold:
        return min(max_k, mid_k)

    return min(max_k, high_k)


def _entropy_to_weight(entropy: float) -> float:
    return 1.0 / (1.0 + max(float(entropy), 0.0))


def _extract_generated_ids_and_scores(
    generation: Any,
    *,
    prompt_len: int,
    include_scores: bool,
):
    if include_scores:
        sequences = generation.sequences
        scores = list(generation.scores or [])
        generated_steps = len(scores)

        if generated_steps > 0:
            if sequences.shape[1] >= prompt_len + generated_steps:
                generated_ids = sequences[:, prompt_len : prompt_len + generated_steps]
            else:
                generated_ids = sequences[:, -generated_steps:]
        else:
            generated_ids = sequences[:, prompt_len:] if sequences.shape[1] > prompt_len else sequences

        return generated_ids, scores

    sequences = generation
    if sequences.shape[1] > prompt_len:
        generated_ids = sequences[:, prompt_len:]
    else:
        generated_ids = sequences

    return generated_ids, []


def _mock_answer(sample: VlmSample) -> str:
    if sample.answer:
        return sample.answer

    if sample.task == "parsing":
        return json.dumps(
            {
                "focused_element": "mock settings",
                "elements": [
                    {
                        "label": "mock icon",
                        "type": "app_icon",
                        "bbox": [0, 0, 100, 100],
                    },
                    {
                        "label": "mock settings",
                        "type": "button",
                        "bbox": [120, 0, 220, 100],
                    },
                ],
            },
            ensure_ascii=False,
        )

    if sample.task == "grounding":
        return json.dumps(
            {
                "label": sample.target_label or "target",
                "type": sample.target_type or "object",
                "bbox": [0, 0, 100, 100],
            },
            ensure_ascii=False,
        )

    return f"mock answer for {sample.task}"
