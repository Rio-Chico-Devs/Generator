"""Worker thread per generazione immagine.

Disaccoppia la pipeline dal main thread UI. Tutte le comunicazioni
passano per signal Qt thread-safe.

Convenzione: ogni .generate() in UI istanzia un nuovo worker, lo .start(),
ascolta i signal. Non si riusa lo stesso worker per multiple generazioni
(semplifica la lifecycle).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from src.core.config import GenerationConfig, GenerationResult
from src.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


class GenerationWorker(QThread):
    # Signals
    progress = pyqtSignal(int, int)  # current_step, total_steps
    image_ready = pyqtSignal(QImage, dict)  # immagine, metadata dict
    error = pyqtSignal(str)  # messaggio errore
    finished_ok = pyqtSignal(list)  # lista GenerationResult

    def __init__(
        self,
        pipeline: Pipeline,
        config: GenerationConfig,
        output_dir: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._config = config
        self._output_dir = output_dir
        self._aborted = False

    def abort(self) -> None:
        """Richiesta di interruzione (al primo step utile)."""
        self._aborted = True

    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as e:
            logger.exception("Errore in GenerationWorker")
            self.error.emit(str(e))

    def _run_impl(self) -> None:
        from PIL.ImageQt import ImageQt

        from src.utils.metadata import (
            build_a1111_parameters_string,
            save_image_with_metadata,
        )
        from src.utils.seed import resolve_seed

        cfg = self._config

        # Risolvi seed prima della generazione (per logging deterministico)
        actual_seed = resolve_seed(cfg.seed)
        cfg.seed = actual_seed

        # Callback progress + abort check
        def on_step(step: int, total: int) -> None:
            if self._aborted:
                # Non c'è un modo pulito di abortire diffusers a metà,
                # ma almeno smettiamo di emettere progress
                raise InterruptedError("Generation aborted by user")
            self.progress.emit(step, total)

        t0 = time.time()

        try:
            images = self._pipeline.generate(cfg, progress_callback=on_step)
        except InterruptedError:
            logger.info("Generazione abortita")
            self.error.emit("Generazione interrotta dall'utente.")
            return

        elapsed = time.time() - t0
        logger.info("Generate completato in %.1fs", elapsed)

        # Salva ogni immagine con metadata
        self._output_dir.mkdir(parents=True, exist_ok=True)

        results: list[GenerationResult] = []
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        for idx, img in enumerate(images):
            timestamp = time.strftime("%H%M%S")
            short_hash = f"{abs(hash((cfg.prompt, actual_seed + idx))) % 0xFFFFFF:06x}"
            base = f"{timestamp}_{short_hash}"
            png_path = self._output_dir / f"{base}.png"
            json_path = self._output_dir / f"{base}.json"

            parameters_text = build_a1111_parameters_string(cfg, actual_seed + idx)
            save_image_with_metadata(img, png_path, parameters_text)

            result = GenerationResult(
                image_path=str(png_path),
                config=cfg,
                actual_seed=actual_seed + idx,
                generation_time_sec=elapsed / max(1, len(images)),
                created_at=now_iso,
            )
            from src.utils.metadata import write_sidecar_json

            write_sidecar_json(json_path, result)
            results.append(result)

            # Notifica UI immagine pronta
            qimg = ImageQt(img.convert("RGBA")).copy()  # .copy() per detach
            self.image_ready.emit(qimg, {"path": str(png_path), "seed": result.actual_seed})

        self.finished_ok.emit(results)
