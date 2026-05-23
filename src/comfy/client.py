"""Client per ComfyUI headless.

Comunica con il processo ComfyUI via HTTP (submit/fetch) e WebSocket
(progress). Include MockComfyClient per sviluppo senza GPU.

Riferimento: docs/COMFY_ENGINE.md
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, Protocol

logger = logging.getLogger(__name__)


class ComfyClientProtocol(Protocol):
    def is_alive(self) -> bool: ...
    def submit(self, workflow: dict) -> str: ...
    def wait_for_completion(
        self, prompt_id: str, progress_callback: Optional[Callable[[int, int], None]]
    ) -> list[Path]: ...
    def interrupt(self) -> None: ...
    def get_vram_usage(self) -> dict: ...


class ComfyClient:
    """Client reale verso ComfyUI."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8188) -> None:
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())
        self._base = f"http://{host}:{port}"

    def is_alive(self) -> bool:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self._base}/system_stats", timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def submit(self, workflow: dict) -> str:
        import urllib.request

        payload = json.dumps(
            {"prompt": workflow, "client_id": self.client_id}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        prompt_id = data["prompt_id"]
        logger.info("Workflow submitted, prompt_id=%s", prompt_id)
        return prompt_id

    def wait_for_completion(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[Path]:
        """Ascolta WebSocket per progress, ritorna path immagini prodotte."""
        import websocket  # websocket-client

        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}")

        try:
            while True:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue
                data = json.loads(msg)
                mtype = data.get("type")

                if mtype == "progress":
                    d = data["data"]
                    if progress_callback:
                        progress_callback(d["value"], d["max"])
                elif mtype == "executing":
                    d = data["data"]
                    if d.get("node") is None and d.get("prompt_id") == prompt_id:
                        break  # esecuzione completata
        finally:
            ws.close()

        return self._fetch_outputs(prompt_id)

    def interrupt(self) -> None:
        import urllib.request

        req = urllib.request.Request(f"{self._base}/interrupt", data=b"")
        urllib.request.urlopen(req, timeout=10)

    def get_vram_usage(self) -> dict:
        """GET /system_stats per la status bar. Ritorna {} se irraggiungibile."""
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self._base}/system_stats", timeout=3) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return {}

    def _fetch_outputs(self, prompt_id: str) -> list[Path]:
        import urllib.request

        from src.utils.paths import get_app_data_dir

        with urllib.request.urlopen(f"{self._base}/history/{prompt_id}", timeout=30) as r:
            history = json.loads(r.read())

        out_dir = get_app_data_dir() / "comfy_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        node_outputs = history.get(prompt_id, {}).get("outputs", {})
        for node_id, node_out in node_outputs.items():
            for img in node_out.get("images", []):
                fname = img["filename"]
                subfolder = img.get("subfolder", "")
                img_type = img.get("type", "output")
                data = self._download_image(fname, subfolder, img_type)
                dest = out_dir / fname
                dest.write_bytes(data)
                paths.append(dest)
        return paths

    def _download_image(self, filename: str, subfolder: str, img_type: str) -> bytes:
        import urllib.parse
        import urllib.request

        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": img_type}
        )
        with urllib.request.urlopen(f"{self._base}/view?{params}", timeout=30) as r:
            return r.read()


class MockComfyClient:
    """Mock per sviluppo UI senza GPU/ComfyUI."""

    def __init__(self, *args, **kwargs) -> None:
        self.client_id = "mock"

    def is_alive(self) -> bool:
        return True

    def submit(self, workflow: dict) -> str:
        return "mock-" + str(uuid.uuid4())[:8]

    def wait_for_completion(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[Path]:
        total = 30
        for i in range(total):
            time.sleep(0.03)
            if progress_callback:
                progress_callback(i + 1, total)
        return [self._make_placeholder(prompt_id)]

    def interrupt(self) -> None:
        pass

    def get_vram_usage(self) -> dict:
        return {
            "devices": [
                {
                    "name": "Mock GPU",
                    "vram_total": 8 * 1024**3,
                    "vram_free": 6 * 1024**3,
                }
            ]
        }

    def _make_placeholder(self, prompt_id: str) -> Path:
        from PIL import Image, ImageDraw

        from src.utils.paths import get_app_data_dir

        out_dir = get_app_data_dir() / "comfy_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prompt_id}.png"

        img = Image.new("RGB", (1024, 1024), (45, 45, 65))
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), f"MOCK OUTPUT\n{prompt_id}", fill=(220, 220, 220))
        img.save(path)
        return path


def make_client(mock: bool, host: str = "127.0.0.1", port: int = 8188):
    """Factory: ritorna client reale o mock."""
    return MockComfyClient() if mock else ComfyClient(host, port)
