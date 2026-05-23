"""Wrapper sopra diffusers per generazione immagini.

Gestisce:
- Lazy loading del modello (caricato solo al primo .generate())
- Switch modello con free VRAM
- LoRA loading/unloading con adapter API
- Modalità mock per sviluppo UI senza GPU

NON gestisce thread: chi chiama .generate() deve essere già in un
QThread o subprocess. Vedi src/workers/generation_worker.py.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from src.core.catalog import CATALOG, VAE_CATALOG, ModelEntry
from src.core.config import GenerationConfig, GenerationResult, LoraSpec
from src.utils.paths import get_models_dir

logger = logging.getLogger(__name__)


# Mapping sampler ID interni → classi scheduler diffusers
# Import differito per non costare al boot
def _resolve_scheduler(sampler_id: str):
    from diffusers import (
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    table = {
        "dpmpp_2m": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++"}),
        "dpmpp_2m_karras": (
            DPMSolverMultistepScheduler,
            {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
        ),
        "euler": (EulerDiscreteScheduler, {}),
        "euler_a": (EulerAncestralDiscreteScheduler, {}),
        "unipc": (UniPCMultistepScheduler, {}),
    }
    cls, kwargs = table.get(sampler_id, table["dpmpp_2m_karras"])
    return cls, kwargs


class Pipeline:
    """Wrapper alto livello su diffusers.

    Stateful: tiene caricato un modello base + i suoi LoRA. Cambiare modello
    base richiede unload completo (chiamare unload() prima).
    """

    def __init__(self, mock: bool = False) -> None:
        self.mock = mock
        self._pipe = None
        self._current_model_id: Optional[str] = None
        self._loaded_loras: dict[str, LoraSpec] = {}
        self._device: str = "cpu"

    # --- Public API --------------------------------------------------

    def load_model(self, model_id: str) -> None:
        """Carica un modello base. Se già caricato, no-op.

        Cambiare modello richiede unload() esplicito prima per liberare VRAM.
        """
        if self.mock:
            self._current_model_id = model_id
            logger.info("[MOCK] load_model(%s)", model_id)
            return

        if model_id == self._current_model_id:
            return

        if self._current_model_id is not None:
            raise RuntimeError(
                f"Modello {self._current_model_id} già caricato. "
                "Chiamare unload() prima di cambiare modello."
            )

        entry = CATALOG.get(model_id)
        if entry is None:
            raise ValueError(f"Modello sconosciuto: {model_id}")

        logger.info("Caricamento modello %s...", model_id)
        t0 = time.time()
        self._pipe = self._build_pipeline(entry)
        self._current_model_id = model_id
        logger.info("Modello caricato in %.1fs", time.time() - t0)

    def unload(self) -> None:
        """Libera VRAM. Necessario prima di cambiare modello o lanciare training."""
        if self.mock:
            self._current_model_id = None
            return

        if self._pipe is not None:
            del self._pipe
            self._pipe = None

        self._loaded_loras.clear()
        self._current_model_id = None

        # Forza garbage collection e cleanup CUDA
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

        logger.info("Pipeline scaricata, VRAM liberata")

    def set_loras(self, loras: list[LoraSpec]) -> None:
        """Imposta i LoRA attivi (carica nuovi, scarica vecchi)."""
        if self.mock:
            self._loaded_loras = {l.adapter_name: l for l in loras}
            return

        if self._pipe is None:
            raise RuntimeError("Caricare prima un modello con load_model()")

        # Strategia semplice: unload tutto, ricarica
        # (l'API multi-adapter di diffusers è più potente ma più fragile)
        try:
            self._pipe.unload_lora_weights()
        except Exception:
            pass  # nessun LoRA caricato

        if not loras:
            self._loaded_loras.clear()
            return

        adapter_names = []
        weights = []
        for spec in loras:
            self._pipe.load_lora_weights(
                spec.path,
                adapter_name=spec.adapter_name,
            )
            adapter_names.append(spec.adapter_name)
            weights.append(spec.weight)

        self._pipe.set_adapters(adapter_names, adapter_weights=weights)
        self._loaded_loras = {l.adapter_name: l for l in loras}
        logger.info("LoRA attivi: %s", [(n, w) for n, w in zip(adapter_names, weights)])

    def generate(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list:
        """Genera immagine/i. Restituisce lista di PIL.Image.

        progress_callback(step, total) viene chiamato a ogni step.
        """
        if self.mock:
            return self._mock_generate(config)

        if self._pipe is None:
            raise RuntimeError("Caricare prima un modello con load_model()")

        import torch

        # Sampler
        scheduler_cls, scheduler_kwargs = _resolve_scheduler(config.sampler)
        self._pipe.scheduler = scheduler_cls.from_config(
            self._pipe.scheduler.config, **scheduler_kwargs
        )

        # Seed
        seed = config.seed if config.seed >= 0 else torch.seed()
        generator = torch.Generator(device=self._device).manual_seed(int(seed))

        # Callback step
        def _cb(pipe_self, step: int, timestep: int, callback_kwargs: dict) -> dict:
            if progress_callback:
                progress_callback(step + 1, config.steps)
            return callback_kwargs

        kwargs = dict(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.steps,
            guidance_scale=config.cfg_scale,
            width=config.width,
            height=config.height,
            num_images_per_prompt=config.batch_size,
            generator=generator,
            callback_on_step_end=_cb,
        )

        result = self._pipe(**kwargs)
        return list(result.images)

    # --- Internal -----------------------------------------------------

    def _build_pipeline(self, entry: ModelEntry):
        import torch
        from diffusers import (
            AutoencoderKL,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
        )

        # Determina device + dtype
        if torch.cuda.is_available():
            self._device = "cuda"
            dtype = torch.float16 if entry.family == "sd15" else torch.float16
            # bf16 sarebbe meglio su SDXL ma fp16 è più ampiamente supportato
        else:
            self._device = "cpu"
            dtype = torch.float32
            logger.warning("CUDA non disponibile, fallback CPU (molto lento)")

        model_path = self._resolve_model_path(entry)

        cls = StableDiffusionXLPipeline if entry.family == "sdxl" else StableDiffusionPipeline

        pipe = cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
            add_watermarker=False,
        )

        # VAE replacement per SDXL fp16
        if entry.requires_vae and entry.requires_vae in VAE_CATALOG:
            vae_entry = VAE_CATALOG[entry.requires_vae]
            vae_path = get_models_dir() / "vae" / vae_entry.filename
            if vae_path.exists():
                vae = AutoencoderKL.from_single_file(str(vae_path), torch_dtype=dtype)
                pipe.vae = vae

        # Ottimizzazioni VRAM (vincolo 8GB)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.debug("xformers non disponibile: %s", e)

        return pipe

    def _resolve_model_path(self, entry: ModelEntry) -> str:
        """Restituisce path locale del modello, o repo_id se da scaricare."""
        local = get_models_dir() / "base" / entry.id
        if local.exists() and (local / "model_index.json").exists():
            return str(local)
        # Fallback al repo_id, lascia che HF gestisca il download
        # (il ModelManager dovrebbe averlo già scaricato però)
        return entry.repo_id

    def _mock_generate(self, config: GenerationConfig) -> list:
        """Genera immagini placeholder per dev UI."""
        from PIL import Image, ImageDraw

        time.sleep(0.5)
        imgs = []
        for i in range(config.batch_size):
            img = Image.new("RGB", (config.width, config.height), color=(40, 40, 60))
            draw = ImageDraw.Draw(img)
            draw.text(
                (20, 20),
                f"MOCK #{i}\n{config.prompt[:60]}\n\n{config.width}x{config.height}",
                fill=(220, 220, 220),
            )
            imgs.append(img)
        return imgs
