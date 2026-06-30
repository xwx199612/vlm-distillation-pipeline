from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.teacher_label_stats import format_teacher_label_summary, summarize_teacher_label_file


def test_summarize_teacher_label_file_reports_unknown_empty_and_schema_counts(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "".join(
            [
                json.dumps(
                    {
                        "id": "row-1",
                        "teacher_answer": {
                            "elements": [
                                {"text": "focused", "type": "unknown", "focused": False},
                                {"text": "", "type": "other", "focused": False},
                            ]
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "id": "row-2",
                        "teacher_answer": '{"elements":[{"text":"Home","type":"tab","focused":true}]}',
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_teacher_label_file(path)

    assert summary["total_samples"] == 2
    assert summary["total_elements"] == 3
    assert summary["unknown_type_ratio"] == 1 / 3
    assert summary["empty_elements_ratio"] == 1 / 3
    assert summary["schema_word_element_count"] == 1


def test_format_teacher_label_summary_emits_expected_fields():
    rendered = format_teacher_label_summary(
        {
            "path": "labels.jsonl",
            "total_samples": 5,
            "total_elements": 20,
            "unknown_type_ratio": 0.25,
            "empty_elements_ratio": 0.10,
            "schema_word_element_count": 3,
        }
    )

    assert "total_samples=5" in rendered
    assert "total_elements=20" in rendered
    assert "unknown_type_ratio=0.2500" in rendered
    assert "empty_elements_ratio=0.1000" in rendered
    assert "schema_word_element_count=3" in rendered
