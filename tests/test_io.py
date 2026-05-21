"""Tests for ``coding.pipeline.io`` JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("polars")

from coding.pipeline.io import append_jsonl  # noqa: E402


def test_append_jsonl_creates_parent_and_appends(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "out.jsonl"

    append_jsonl(out, {"event_id": "n1", "labels": ["A01"]})
    append_jsonl(out, {"event_id": "n2", "labels": ["B02", "C03"]})

    assert out.exists(), "append_jsonl should create the file (and parent dirs)"

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first == {"event_id": "n1", "labels": ["A01"]}
    assert second == {"event_id": "n2", "labels": ["B02", "C03"]}


def test_append_jsonl_preserves_unicode(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    append_jsonl(out, {"note": "α-thalassaemia"})
    line = out.read_text(encoding="utf-8").strip()
    # ensure_ascii=False is set in the implementation; the original glyph
    # should survive a round-trip.
    assert "α-thalassaemia" in line
    assert json.loads(line)["note"] == "α-thalassaemia"
