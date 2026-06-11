"""Tests per src/training/output_stream.py — split su \\r e \\n (tqdm)."""
from __future__ import annotations

import io

from src.training.output_stream import iter_output_lines


def _lines(text: str) -> list[str]:
    return list(iter_output_lines(io.StringIO(text)))


def test_empty_stream():
    assert _lines("") == []


def test_single_line_with_newline():
    assert _lines("hello\n") == ["hello"]


def test_single_line_no_trailing_newline():
    # Resto nel buffer a EOF: va consegnato
    assert _lines("hello") == ["hello"]


def test_multiple_newlines():
    assert _lines("a\nb\nc\n") == ["a", "b", "c"]


def test_carriage_return_splits():
    # Comportamento tqdm: aggiornamenti sulla stessa riga via \r
    assert _lines("step 1\rstep 2\rstep 3\n") == ["step 1", "step 2", "step 3"]


def test_crlf_no_empty_lines():
    # \r\n non deve produrre righe vuote
    assert _lines("a\r\nb\r\n") == ["a", "b"]


def test_consecutive_separators_skipped():
    assert _lines("a\n\n\nb\n") == ["a", "b"]


def test_tqdm_progress_realistic():
    # Simulazione output tqdm di sd-scripts
    text = (
        "steps:   3%|##        | 10/300 [00:05<02:30, 1.9it/s, avr_loss=0.123]\r"
        "steps:   6%|####      | 20/300 [00:10<02:20, 2.0it/s, avr_loss=0.119]\r"
        "epoch 1/10\n"
    )
    out = _lines(text)
    assert len(out) == 3
    assert "10/300" in out[0]
    assert "20/300" in out[1]
    assert out[2] == "epoch 1/10"


def test_mixed_cr_and_lf():
    assert _lines("a\rb\nc\rd") == ["a", "b", "c", "d"]
