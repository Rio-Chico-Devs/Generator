from __future__ import annotations

from src.comfy.client import ComfyClient, MockComfyClient, make_client


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
