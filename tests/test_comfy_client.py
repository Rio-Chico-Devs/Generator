from __future__ import annotations

import json
from pathlib import Path

from src.comfy.client import (
    ComfyClient,
    ComfyError,
    ComfyOutOfMemory,
    MockComfyClient,
    _format_submit_error,
    _looks_like_oom,
    make_client,
)


def test_make_client_returns_mock_when_mock_true():
    assert isinstance(make_client(mock=True), MockComfyClient)


def test_make_client_returns_real_when_mock_false():
    c = make_client(mock=False, port=12345)
    assert isinstance(c, ComfyClient)
    assert c.port == 12345


def test_mock_is_alive_and_submit():
    c = MockComfyClient()
    assert c.is_alive() is True
    pid = c.submit({"any": "graph"})
    assert isinstance(pid, str)
    assert pid.startswith("mock-")


def test_mock_get_vram_usage_shape():
    stats = MockComfyClient().get_vram_usage()
    assert "devices" in stats
    assert stats["devices"][0]["vram_total"] > 0


def test_mock_progress_callback_is_called():
    c = MockComfyClient()
    calls: list[tuple[int, int]] = []
    # wait_for_completion emette progress prima di generare il placeholder
    # (che richiede Pillow); cattura ImportError se Pillow non c'è.
    try:
        c.wait_for_completion("mock-x", progress_callback=lambda v, m: calls.append((v, m)))
    except ImportError:
        pass
    assert calls
    assert calls[-1][0] == calls[-1][1]  # ultimo step == totale


# ---------------------------------------------------------------------------
# Robustezza: parsing errori, OOM detection, build_execution_error
# ---------------------------------------------------------------------------


def test_format_submit_error_extracts_error_message():
    body = '{"error": {"message": "Prompt has no outputs", "details": "node 9"}}'
    out = _format_submit_error(body)
    assert "Prompt has no outputs" in out
    assert "node 9" in out


def test_format_submit_error_extracts_node_errors():
    body = (
        '{"node_errors": {"10": {"class_type": "LoraLoader", '
        '"errors": [{"message": "lora_name non trovato"}]}}}'
    )
    out = _format_submit_error(body)
    assert "LoraLoader" in out
    assert "lora_name non trovato" in out


def test_format_submit_error_plaintext_passthrough():
    assert _format_submit_error("boom plain") == "boom plain"


def test_format_submit_error_empty():
    assert _format_submit_error("") == ""


def test_looks_like_oom_detects_cuda_oom():
    assert _looks_like_oom("CUDA out of memory") is True
    assert _looks_like_oom("torch.cuda.OutOfMemoryError: ...") is True
    assert _looks_like_oom("ValueError: bad shape") is False


def test_build_execution_error_oom():
    err = ComfyClient._build_execution_error({
        "node_type": "KSampler",
        "exception_type": "torch.cuda.OutOfMemoryError",
        "exception_message": "CUDA out of memory. Tried to allocate ...",
    })
    assert isinstance(err, ComfyOutOfMemory)
    assert "lowvram" in str(err)


def test_build_execution_error_generic():
    err = ComfyClient._build_execution_error({
        "node_type": "VAEDecode",
        "exception_type": "ValueError",
        "exception_message": "tensor shape mismatch",
    })
    assert isinstance(err, ComfyError)
    assert not isinstance(err, ComfyOutOfMemory)
    assert "VAEDecode" in str(err)
    assert "tensor shape mismatch" in str(err)


# ---------------------------------------------------------------------------
# Sicurezza: _fetch_outputs non deve scrivere fuori da out_dir
# ---------------------------------------------------------------------------


class _FakeResp:
    """Context manager minimale che simula la risposta di urlopen."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def _patch_history(monkeypatch, history: dict) -> None:
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(json.dumps(history).encode("utf-8")),
    )


def test_fetch_outputs_sanitizes_path_traversal(tmp_path, monkeypatch):
    from src.utils import paths as paths_mod

    # out_dir isolato sotto tmp_path
    monkeypatch.setattr(paths_mod, "get_app_data_dir", lambda: tmp_path)
    import src.comfy.client as client_mod
    monkeypatch.setattr(client_mod, "ComfyClient", ComfyClient)

    c = ComfyClient(port=9999)
    # Evita la seconda chiamata di rete: _download_image ritorna byte fissi.
    monkeypatch.setattr(c, "_download_image", lambda f, s, t: b"PNGDATA")

    pid = "p1"
    _patch_history(monkeypatch, {
        pid: {"outputs": {"9": {"images": [
            {"filename": "../../evil.png", "subfolder": "", "type": "output"},
        ]}}},
    })

    out_dir = tmp_path / "comfy_outputs"
    paths = c._fetch_outputs(pid)

    assert len(paths) == 1
    written = paths[0]
    # Il file deve stare DENTRO out_dir, non fuori (nessun escape via ../).
    assert written.parent == out_dir
    assert written.name == "evil.png"
    assert written.read_bytes() == b"PNGDATA"
    # Nessun file scritto fuori da out_dir.
    assert not (tmp_path.parent / "evil.png").exists()


def test_fetch_outputs_skips_empty_filename(tmp_path, monkeypatch):
    from src.utils import paths as paths_mod

    monkeypatch.setattr(paths_mod, "get_app_data_dir", lambda: tmp_path)

    c = ComfyClient(port=9999)
    monkeypatch.setattr(c, "_download_image", lambda f, s, t: b"X")

    pid = "p2"
    _patch_history(monkeypatch, {
        pid: {"outputs": {"9": {"images": [
            {"filename": "/", "subfolder": "", "type": "output"},
        ]}}},
    })

    paths = c._fetch_outputs(pid)
    assert paths == []
