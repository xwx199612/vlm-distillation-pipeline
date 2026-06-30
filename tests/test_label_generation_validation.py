from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.data_manifest import VlmSample
from vlm_distill.teacher_validation import validate_teacher_output_file
from vlm_distill.stage_answer_labeling import (
    _label_sample,
    _normalize_teacher_answer,
)


class _TokenizingTeacher:
    def __init__(self, answer: str):
        self._answer = answer

    def answer(self, sample: VlmSample) -> dict:
        return {
            "teacher_answer": self._answer,
            "teacher_tokens": [999],
            "teacher_confidence": 1.0,
            "teacher_rationale": "test",
        }

    def tokenize_teacher_answer(self, answer: str) -> list[int]:
        return [ord(char) for char in answer]

    def decode_teacher_tokens(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class _Config:
    class distillation:
        min_teacher_confidence = 0.0


def _sample() -> VlmSample:
    return VlmSample(
        id="screen-1",
        image="screen.jpg",
        task="parsing",
        query="List all visible UI elements.",
    )


def test_object_list_teacher_answer_is_preserved():
    raw_answer = json.dumps(
        {
            "elements": [
                {"text": "Search", "type": "tab", "focused": False},
                {"text": "Home", "type": "tab", "focused": True},
            ]
        }
    )

    normalized = json.loads(_normalize_teacher_answer(_sample(), raw_answer))

    assert normalized == {
        "elements": [
            {"focused": False, "text": "Search", "type": "tab"},
            {"focused": True, "text": "Home", "type": "tab"},
        ]
    }


def test_string_list_teacher_answer_is_converted_to_object_list():
    raw_answer = '{"elements":["Search","Home"]}'

    normalized = json.loads(_normalize_teacher_answer(_sample(), raw_answer))

    assert normalized == {
        "elements": [
            {"focused": False, "text": "Search", "type": "input"},
            {"focused": False, "text": "Home", "type": "tab"},
        ]
    }


def test_teacher_tokens_are_recomputed_after_normalization():
    row = _label_sample(
        _Config(),
        _TokenizingTeacher('{"elements":["Search"]}'),
        _sample(),
    )

    assert row is not None
    assert json.loads(row["teacher_answer"]) == {
        "elements": [{"focused": False, "text": "Search", "type": "input"}]
    }
    assert row["teacher_tokens"] == [ord(char) for char in row["teacher_answer"]]
    assert row["teacher_tokens"] != [999]


def test_validate_teacher_output_file_checks_decoded_tokens_against_final_answer(tmp_path: Path):
    answer = _normalize_teacher_answer(
        _sample(),
        '{"elements":[{"text":"Search","type":"input","focused":false}]}',
    )
    label_path = tmp_path / "labels.jsonl"
    label_path.write_text(
        json.dumps(
            {
                "id": "screen-1",
                "image": "screen.jpg",
                "query": "List all visible UI elements.",
                "teacher_answer": answer,
                "teacher_tokens": [ord(char) for char in answer],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_teacher_output_file(
        label_path,
        decode_tokens=lambda tokens: "".join(chr(token) for token in tokens),
    )

    assert summary["valid_json_rows"] == 1
    assert summary["schema_valid_rows"] == 1
    assert summary["answer_token_mismatch_rows"] == 0


def test_validate_teacher_output_file_ignores_im_end_in_decode_comparison(tmp_path: Path):
    answer = _normalize_teacher_answer(
        _sample(),
        '{"elements":[{"text":"Home","type":"tab","focused":true}]}',
    )
    label_path = tmp_path / "labels.jsonl"
    label_path.write_text(
        json.dumps(
            {
                "id": "screen-1",
                "image": "screen.jpg",
                "query": "List all visible UI elements.",
                "teacher_answer": answer,
                "teacher_tokens": [1, 2, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_teacher_output_file(
        label_path,
        decode_tokens=lambda _tokens: answer + "<|im_end|>",
    )

    assert summary["answer_token_mismatch_rows"] == 0


def test_schema_word_elements_are_dropped_and_unknown_types_are_repaired():
    raw_answer = json.dumps(
        {
            "elements": [
                {"text": "focused", "type": "unknown", "focused": False},
                {"text": "Home", "type": "unknown", "focused": True},
                {"text": "Details", "type": "unknown", "focused": False},
                {"text": "Spotify", "type": "unknown", "focused": False},
                {"text": "Program Guide", "type": "unknown", "focused": False},
                {"text": "Recommended Movies", "type": "unknown", "focused": False},
                {"text": "Mystery Label", "type": "unknown", "focused": False},
            ]
        }
    )

    normalized = json.loads(_normalize_teacher_answer(_sample(), raw_answer))

    assert normalized == {
        "elements": [
            {"focused": True, "text": "Home", "type": "tab"},
            {"focused": False, "text": "Details", "type": "button"},
            {"focused": False, "text": "Spotify", "type": "app_icon"},
            {"focused": False, "text": "Program Guide", "type": "menu_item"},
            {"focused": False, "text": "Recommended Movies", "type": "tile"},
            {"focused": False, "text": "Mystery Label", "type": "other"},
        ]
    }


def test_fallback_string_labels_use_repaired_types_not_unknown():
    raw_answer = '{"elements":["focused","Search","Netflix","+","Channel Setup"]}'

    normalized = json.loads(_normalize_teacher_answer(_sample(), raw_answer))

    assert normalized == {
        "elements": [
            {"focused": False, "text": "Search", "type": "input"},
            {"focused": False, "text": "Netflix", "type": "app_icon"},
            {"focused": False, "text": "+", "type": "button"},
            {"focused": False, "text": "Channel Setup", "type": "menu_item"},
        ]
    }
