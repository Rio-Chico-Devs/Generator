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


# --- Eccezioni tipizzate ---------------------------------------------------


class ComfyError(RuntimeError):
    """Errore generico lato ComfyUI (submit rifiutato, esecuzione fallita)."""


class ComfyInterrupted(ComfyError):
    """L'esecuzione è stata interrotta su richiesta (interrupt())."""


class ComfyOutOfMemory(ComfyError):
    """ComfyUI è andato in out-of-memory (VRAM insufficiente)."""


_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda out of memory",
    "allocation on device",
    "alloc_pool",
)


def _looks_like_oom(text: str) -> bool:
    t = (text or "").lower()
    return any(marker in t for marker in _OOM_MARKERS)


def _format_submit_error(body: str) -> str:
    """Estrae un messaggio leggibile dal corpo d'errore di /prompt.

    ComfyUI risponde con ``{"error": {...}, "node_errors": {...}}``: senza
    questo parsing l'utente vedrebbe solo "HTTP Error 400".
    """
    if not body:
        return ""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body.strip()[:300]

    parts: list[str] = []
    err = data.get("error")
    if isinstance(err, dict) and err.get("message"):
        msg = err["message"]
        details = err.get("details")
        parts.append(f"{msg} ({details})" if details else str(msg))
    elif isinstance(err, str):
        parts.append(err)

    node_errors = data.get("node_errors") or {}
    for node_id, info in node_errors.items():
        klass = info.get("class_type", "?") if isinstance(info, dict) else "?"
        for e in (info.get("errors", []) if isinstance(info, dict) else []):
            parts.append(f"[nodo {node_id} {klass}] {e.get('message', '')}".strip())

    return " · ".join(p for p in parts if p)


class ComfyClientProtocol(Protocol):
    def is_alive(self) -> bool: ...
    def submit(self, workflow: dict) -> str: ...
    def wait_for_completion(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> list[Path]: ...
    def interrupt(self) -> None: ...
    def clear_queue(self) -> None: ...
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
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {"prompt": workflow, "client_id": self.client_id}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # ComfyUI mette i dettagli della validazione nel corpo (HTTP 400):
            # senza leggerli l'utente vedrebbe solo "HTTP Error 400".
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:
                pass
            raise ComfyError(
                f"ComfyUI ha rifiutato il workflow: {_format_submit_error(body) or e}"
            ) from e
        except urllib.error.URLError as e:
            raise ComfyError(f"ComfyUI non raggiungibile: {e.reason}") from e

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            # Validazione fallita ma con 200: node_errors nel corpo.
            raise ComfyError(
                f"Workflow non valido: {_format_submit_error(json.dumps(data)) or data}"
            )
        logger.info("Workflow submitted, prompt_id=%s", prompt_id)
        return prompt_id

    # Timeout di ogni singola recv(): breve per permettere l'aborto rapido.
    # Il watchdog dell'idle totale usa _IDLE_WATCHDOG_SEC.
    _RECV_TIMEOUT_SEC: float = 2.0
    _IDLE_WATCHDOG_SEC: float = 120.0

    def wait_for_completion(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> list[Path]:
        """Ascolta il WebSocket per il progresso, ritorna le immagini prodotte.

        Robustezza:
        - execution_error / execution_interrupted → eccezioni tipizzate;
        - silenzio per _IDLE_WATCHDOG_SEC → watchdog; se ComfyUI è vivo ma
          silenzioso (nodo lungo), continua; se crashato, solleva;
        - abort_check() → controlla aborto a ogni ciclo recv (timeout 2s),
          così stop() non deve aspettare 120s per sbloccare il thread.
        """
        import websocket  # websocket-client

        ws = websocket.WebSocket()
        ws.settimeout(self._RECV_TIMEOUT_SEC)
        try:
            ws.connect(f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}")
        except (OSError, websocket.WebSocketException) as e:
            raise ComfyError(f"Connessione WebSocket a ComfyUI fallita: {e}") from e

        idle_since = time.monotonic()

        try:
            while True:
                # Controlla aborto PRIMA di bloccarsi in recv.
                if abort_check is not None and abort_check():
                    raise ComfyInterrupted("Esecuzione interrotta dall'utente.")

                try:
                    msg = ws.recv()
                    idle_since = time.monotonic()  # qualsiasi messaggio resetta l'idle
                except websocket.WebSocketTimeoutException:
                    # Controlla aborto anche qui: dopo ogni recv timeout di 2s.
                    if abort_check is not None and abort_check():
                        raise ComfyInterrupted("Esecuzione interrotta dall'utente.")
                    idle_elapsed = time.monotonic() - idle_since
                    if idle_elapsed >= self._IDLE_WATCHDOG_SEC:
                        if self._is_finished(prompt_id):
                            break  # completato, messaggio finale perso
                        if not self.is_alive():
                            raise ComfyError(
                                "ComfyUI non risponde più (possibile crash). "
                                "Controlla logs/comfyui.log."
                            )
                        idle_since = time.monotonic()  # resetta per evitare re-trigger
                    continue
                except websocket.WebSocketConnectionClosedException:
                    if self._is_finished(prompt_id):
                        break
                    raise ComfyError(
                        "Connessione a ComfyUI chiusa durante l'esecuzione "
                        "(possibile crash). Controlla logs/comfyui.log."
                    )

                if not isinstance(msg, str):
                    continue  # frame binario (anteprima) — ignora
                data = json.loads(msg)
                mtype = data.get("type")
                d = data.get("data") or {}

                if mtype == "progress":
                    if progress_callback and "value" in d and "max" in d:
                        # Cast esplicito: JSON può dare float (es. 1.0 invece di 1)
                        progress_callback(int(d["value"]), int(d["max"]))
                elif mtype == "executing":
                    if d.get("node") is None and d.get("prompt_id") == prompt_id:
                        break  # esecuzione completata
                elif mtype == "execution_interrupted" and d.get("prompt_id") == prompt_id:
                    raise ComfyInterrupted("Esecuzione interrotta.")
                elif mtype == "execution_error" and d.get("prompt_id") == prompt_id:
                    raise self._build_execution_error(d)
        finally:
            try:
                ws.close()
            except Exception:
                pass

        outputs = self._fetch_outputs(prompt_id)
        if not outputs:
            raise ComfyError(
                f"ComfyUI ha completato il prompt {prompt_id} ma non ha prodotto "
                "immagini. Controlla il workflow e i nodi SaveImage."
            )
        return outputs

    @staticmethod
    def _build_execution_error(d: dict) -> ComfyError:
        node_type = d.get("node_type", "?")
        exc_msg = d.get("exception_message", "errore sconosciuto")
        exc_type = d.get("exception_type", "")
        if _looks_like_oom(f"{exc_type} {exc_msg}"):
            return ComfyOutOfMemory(
                f"Memoria GPU insufficiente (OOM) sul nodo '{node_type}'. "
                "Prova a ridurre la risoluzione, generare una sola immagine "
                "per volta, o impostare comfy_vram_mode='lowvram'."
            )
        return ComfyError(f"Esecuzione fallita sul nodo '{node_type}': {exc_msg}")

    def _is_finished(self, prompt_id: str) -> bool:
        """True se il prompt risulta completato in history."""
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self._base}/history/{prompt_id}", timeout=10
            ) as r:
                hist = json.loads(r.read())
            return prompt_id in hist
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def interrupt(self) -> None:
        import urllib.request

        req = urllib.request.Request(f"{self._base}/interrupt", data=b"")
        with urllib.request.urlopen(req, timeout=10):
            pass

    def clear_queue(self) -> None:
        """Svuota la coda pending di ComfyUI (POST /queue {"clear": true})."""
        import urllib.request

        payload = json.dumps({"clear": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/queue",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.warning("clear_queue() HTTP error: %s", exc)

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
                # Sanitizza il nome per la scrittura locale: il filename arriva da
                # una risposta di rete, .name scarta qualsiasi componente di path
                # (es. "../") impedendo scritture fuori da out_dir.
                safe_name = Path(fname).name
                if not safe_name:
                    logger.warning("Output con filename non valido ignorato: %r", fname)
                    continue
                data = self._download_image(fname, subfolder, img_type)
                dest = out_dir / safe_name
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
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> list[Path]:
        total = 30
        for i in range(total):
            time.sleep(0.03)
            if abort_check is not None and abort_check():
                raise ComfyInterrupted("Esecuzione interrotta dall'utente.")
            if progress_callback:
                progress_callback(i + 1, total)
        return [self._make_placeholder(prompt_id)]

    def interrupt(self) -> None:
        pass

    def clear_queue(self) -> None:
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
