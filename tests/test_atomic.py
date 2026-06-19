"""Test per src/utils/atomic — scritture atomiche temp+fsync+rename.

Copre tutti e tre i primitivi: atomic_write_text, atomic_write_bytes,
atomic_append_line. Nessuna dipendenza esterna (solo stdlib).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.utils.atomic import atomic_append_line, atomic_write_bytes, atomic_write_text


# ---------------------------------------------------------------------------
# atomic_write_text
# ---------------------------------------------------------------------------


def test_write_text_roundtrip(tmp_path):
    p = tmp_path / "file.txt"
    atomic_write_text(p, "ciao mondo")
    assert p.read_text(encoding="utf-8") == "ciao mondo"


def test_write_text_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "file.txt"
    atomic_write_text(p, "ok")
    assert p.exists()
    assert p.read_text(encoding="utf-8") == "ok"


def test_write_text_overwrites_existing(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("vecchio", encoding="utf-8")
    atomic_write_text(p, "nuovo")
    assert p.read_text(encoding="utf-8") == "nuovo"


def test_write_text_no_tmp_file_left_on_success(tmp_path):
    p = tmp_path / "file.txt"
    atomic_write_text(p, "data")
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_write_text_empty_string(tmp_path):
    p = tmp_path / "empty.txt"
    atomic_write_text(p, "")
    assert p.read_text(encoding="utf-8") == ""
    assert p.exists()


def test_write_text_unicode_content(tmp_path):
    p = tmp_path / "unicode.txt"
    content = "日本語テスト\nè é ü ñ 🎨"
    atomic_write_text(p, content)
    assert p.read_text(encoding="utf-8") == content


def test_write_text_multiline(tmp_path):
    p = tmp_path / "multi.txt"
    lines = "\n".join(f"linea {i}" for i in range(100))
    atomic_write_text(p, lines)
    assert p.read_text(encoding="utf-8") == lines


def test_write_text_json_roundtrip(tmp_path):
    p = tmp_path / "data.json"
    obj = {"key": [1, 2, 3], "nested": {"a": True, "b": None}}
    atomic_write_text(p, json.dumps(obj, indent=2))
    assert json.loads(p.read_text(encoding="utf-8")) == obj


def test_write_text_multiple_overwrites(tmp_path):
    p = tmp_path / "file.txt"
    for i in range(5):
        atomic_write_text(p, f"version-{i}")
    assert p.read_text(encoding="utf-8") == "version-4"


def test_write_text_idempotent(tmp_path):
    p = tmp_path / "file.txt"
    content = "identico"
    atomic_write_text(p, content)
    atomic_write_text(p, content)
    assert p.read_text(encoding="utf-8") == content


def test_write_text_large_payload(tmp_path):
    p = tmp_path / "large.txt"
    content = "x" * (1024 * 1024)  # 1 MB
    atomic_write_text(p, content)
    assert p.read_text(encoding="utf-8") == content


def test_write_text_preserves_exact_whitespace(tmp_path):
    p = tmp_path / "ws.txt"
    content = "  leading\ntrailing  \n\ttab\n"
    atomic_write_text(p, content)
    assert p.read_text(encoding="utf-8") == content


def test_write_text_no_extra_tmp_files(tmp_path):
    p = tmp_path / "out.txt"
    for i in range(10):
        atomic_write_text(p, str(i))
    # Solo il file finale, nessun .tmp rimasto
    assert sorted(tmp_path.iterdir()) == [p]


# ---------------------------------------------------------------------------
# atomic_write_bytes
# ---------------------------------------------------------------------------


def test_write_bytes_roundtrip(tmp_path):
    p = tmp_path / "file.bin"
    data = b"\x00\x01\x02\xff\xfe"
    atomic_write_bytes(p, data)
    assert p.read_bytes() == data


def test_write_bytes_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "c.bin"
    atomic_write_bytes(p, b"test")
    assert p.exists()
    assert p.read_bytes() == b"test"


def test_write_bytes_empty(tmp_path):
    p = tmp_path / "empty.bin"
    atomic_write_bytes(p, b"")
    assert p.read_bytes() == b""
    assert p.exists()


def test_write_bytes_null_bytes_preserved(tmp_path):
    p = tmp_path / "nulls.bin"
    data = bytes(256)  # 256 null bytes
    atomic_write_bytes(p, data)
    assert p.read_bytes() == data


def test_write_bytes_large_data(tmp_path):
    p = tmp_path / "large.bin"
    data = os.urandom(1024 * 1024)  # 1 MB
    atomic_write_bytes(p, data)
    assert p.read_bytes() == data


def test_write_bytes_no_tmp_left_on_success(tmp_path):
    p = tmp_path / "file.bin"
    atomic_write_bytes(p, b"data")
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_write_bytes_overwrites_existing(tmp_path):
    p = tmp_path / "file.bin"
    p.write_bytes(b"old")
    atomic_write_bytes(p, b"new data")
    assert p.read_bytes() == b"new data"


def test_write_bytes_png_magic_preserved(tmp_path):
    """I magic byte PNG devono sopravvivere intatti."""
    png_magic = b"\x89PNG\r\n\x1a\n" + bytes(100)
    p = tmp_path / "image.png"
    atomic_write_bytes(p, png_magic)
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_bytes_all_byte_values(tmp_path):
    p = tmp_path / "all_bytes.bin"
    data = bytes(range(256))
    atomic_write_bytes(p, data)
    assert p.read_bytes() == data


def test_write_bytes_multiple_overwrites(tmp_path):
    p = tmp_path / "file.bin"
    for i in range(5):
        atomic_write_bytes(p, bytes([i] * 10))
    assert p.read_bytes() == bytes([4] * 10)


# ---------------------------------------------------------------------------
# atomic_append_line
# ---------------------------------------------------------------------------


def test_append_line_creates_file(tmp_path):
    p = tmp_path / "log.jsonl"
    atomic_append_line(p, '{"a": 1}')
    assert p.exists()


def test_append_line_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "diary.jsonl"
    atomic_append_line(p, '{"x": 1}')
    assert p.exists()


def test_append_line_single_entry(tmp_path):
    p = tmp_path / "log.jsonl"
    atomic_append_line(p, '{"entry": 1}')
    content = p.read_text(encoding="utf-8")
    assert content.strip() == '{"entry": 1}'
    assert content.endswith("\n")


def test_append_line_multiple_entries(tmp_path):
    p = tmp_path / "log.jsonl"
    for i in range(5):
        atomic_append_line(p, f'{{"n": {i}}}')
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        assert json.loads(line)["n"] == i


def test_append_line_adds_newline_if_missing(tmp_path):
    p = tmp_path / "log.jsonl"
    atomic_append_line(p, "no-newline")
    assert p.read_text(encoding="utf-8") == "no-newline\n"


def test_append_line_does_not_double_newline(tmp_path):
    p = tmp_path / "log.jsonl"
    atomic_append_line(p, "already\n")
    content = p.read_text(encoding="utf-8")
    assert content == "already\n"
    assert content.count("\n") == 1


def test_append_line_appends_to_existing(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text("first\n", encoding="utf-8")
    atomic_append_line(p, "second")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines == ["first", "second"]


def test_append_line_unicode(tmp_path):
    p = tmp_path / "log.jsonl"
    entry = '{"prompt": "iris, 日本語, 🎨"}'
    atomic_append_line(p, entry)
    assert p.read_text(encoding="utf-8").strip() == entry


def test_append_line_preserves_order(tmp_path):
    p = tmp_path / "log.jsonl"
    n = 50
    for i in range(n):
        atomic_append_line(p, str(i))
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines == [str(i) for i in range(n)]


def test_append_line_empty_string(tmp_path):
    p = tmp_path / "log.jsonl"
    atomic_append_line(p, "")
    assert p.read_text(encoding="utf-8") == "\n"


def test_append_line_json_roundtrip(tmp_path):
    p = tmp_path / "diary.jsonl"
    entries = [{"id": i, "step": f"step_{i}"} for i in range(10)]
    for e in entries:
        atomic_append_line(p, json.dumps(e, ensure_ascii=False))
    loaded = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert loaded == entries


def test_append_line_file_size_grows(tmp_path):
    p = tmp_path / "log.jsonl"
    for i in range(10):
        prev_size = p.stat().st_size if p.exists() else 0
        atomic_append_line(p, f"line {i}")
        assert p.stat().st_size > prev_size
