from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
from dataclasses import asdict
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin

from .config_schema import PipelineConfig
from .data_manifest import VlmSample, write_jsonl


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
        elif sample.task == "screen_parsing":
            teacher_answer = json.dumps(
                {
                    "focused_element": "mock settings",
                    "selectable_elements": ["mock icon", "mock settings"],
                },
                ensure_ascii=False,
            )
        elif sample.task == "grounding":
            teacher_answer = json.dumps(
                {
                    "label": sample.target_label or "target",
                    "bbox": sample.bbox or [0, 0, 100, 100],
                },
                ensure_ascii=False,
            )
        else:
            teacher_answer = f"mock answer for {sample.task}"

        return {
            "teacher_answer": teacher_answer,
            "teacher_confidence": round(confidence, 4),
            "teacher_rationale": "Mock backend used for pipeline validation.",
        }


class HuggingFaceTeacher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        try:
            import torch
            from transformers import (
                AutoModelForVision2Seq,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers and bitsandbytes to use the Hugging Face teacher backend."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            config.teacher.model_name,
            trust_remote_code=True,
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

        self.model = AutoModelForVision2Seq.from_pretrained(
            config.teacher.model_name,
            **model_kwargs,
        )

    def answer(self, sample: VlmSample) -> dict:
        from PIL import Image

        image_path = self.config.data.image_root / sample.image
        image = Image.open(image_path).convert("RGB")

        prompt = _format_prompt(self.config, sample)

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

        output_ids = self.model.generate(
            **inputs,
            do_sample=self.config.teacher.temperature > 0,
            temperature=self.config.teacher.temperature if self.config.teacher.temperature > 0 else None,
            max_new_tokens=self.config.teacher.max_new_tokens,
        )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "teacher_answer": answer.strip(),
            "teacher_confidence": 1.0,
            "teacher_rationale": "Generated by Hugging Face teacher backend.",
        }

class OpenAICompatibleTeacher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._requests = None
        self._openai_client = None
        if not config.teacher.base_url:
            raise ValueError("teacher.base_url is required when backend='openai_compatible'.")

    def answer(self, sample: VlmSample) -> dict:
        image_path = self.config.data.image_root / sample.image
        image_data_url = _image_to_data_url(image_path)
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
        image_data = _image_to_base64(image_path)
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
    return config.distillation.prompt_template.format(
        query=sample.query,
        target_label=sample.target_label or "target object",
        task=sample.task,
    )


def _target_from_existing_annotation(sample: VlmSample) -> str | None:
    if sample.task == "screen_parsing" and sample.elements:
        return json.dumps(
            {
                "focused_element": "mock settings",
                "selectable_elements": ["mock icon", "mock settings"],
            },
            ensure_ascii=False,
        )

    if sample.task == "grounding" and sample.bbox:
        return json.dumps(
            {
                "label": sample.target_label or "target object",
                "bbox": sample.bbox,
            },
            ensure_ascii=False,
        )

    if sample.answer:
        return sample.answer

    return None

def create_distillation_dataset(config: PipelineConfig, samples: list[VlmSample]) -> Path:
    teacher = build_teacher(config)
    rows: list[dict] = []
    for sample in samples:
        existing_target = _target_from_existing_annotation(sample)

        if existing_target is not None:
            label = {
                "teacher_answer": existing_target,
                "teacher_confidence": 1.0,
                "teacher_rationale": "Used existing manifest annotation.",
            }
        else:
            label = teacher.answer(sample)

        if label["teacher_confidence"] < config.distillation.min_teacher_confidence:
            continue

        rows.append(
            {
                **asdict(sample),
                **label,
                config.distillation.target_field: label["teacher_answer"],
            }
        )
    write_jsonl(config.data.distill_path, rows)
    return config.data.distill_path


def _image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{_image_to_base64(image_path)}"


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