from __future__ import annotations

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
