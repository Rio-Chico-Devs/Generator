"""Logica della schermata di generazione, separata da Qt.

Tutto ciò che la GenerateView calcola senza toccare widget vive qui, così è
testabile senza PyQt6: conteggio token approssimato, preset dimensioni,
liste sampler/scheduler, costruzione del dict di parametri per il
RecipeWorker e stima costo in crediti.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.core.config import GenerationConfig, LoraSpec
from src.core.credits import CostBreakdown, CostModel, estimate_cost


# ---------------------------------------------------------------------------
# Conteggio token (approssimato)
# ---------------------------------------------------------------------------

# CLIP usa una finestra di 75 token utili per chunk. Oltre, A1111/ComfyUI
# spezzano in chunk successivi: non è un errore, ma è utile avvisare.
TOKEN_LIMIT = 75
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def approx_token_count(text: str) -> int:
    """Stima approssimata dei token CLIP di un prompt.

    Non è il tokenizer BPE reale (che richiederebbe transformers): conta
    parole e segni di punteggiatura come token separati. Tende a sovrastimare
    leggermente, quindi è prudente come indicatore di "stai per superare il
    limite".
    """
    if not text or not text.strip():
        return 0
    return len(_TOKEN_RE.findall(text))


@dataclass(frozen=True)
class TokenStatus:
    count: int
    limit: int = TOKEN_LIMIT

    @property
    def level(self) -> str:
        """ok | warn | over — per colorare il contatore nella UI."""
        if self.count <= self.limit:
            return "ok"
        if self.count <= self.limit * 2:
            return "warn"
        return "over"

    @property
    def chunks(self) -> int:
        """Quanti chunk CLIP da 75 token occupa il prompt (minimo 1)."""
        if self.count == 0:
            return 0
        return (self.count + self.limit - 1) // self.limit

    def label(self) -> str:
        return f"{self.count} / {self.limit} token"


def token_status(text: str) -> TokenStatus:
    return TokenStatus(count=approx_token_count(text))


# ---------------------------------------------------------------------------
# Preset dimensioni e liste sampler/scheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizePreset:
    key: str
    label: str
    width: int
    height: int


# Rapporti nativi SDXL (~1 megapixel), quelli su cui i modelli rendono meglio.
SIZE_PRESETS: tuple[SizePreset, ...] = (
    SizePreset("portrait", "Ritratto", 832, 1216),
    SizePreset("landscape", "Panorama", 1216, 832),
    SizePreset("square", "Quadrato", 1024, 1024),
)


def preset_for(width: int, height: int) -> Optional[SizePreset]:
    """Ritorna il preset che combacia con le dimensioni date, o None (custom)."""
    for p in SIZE_PRESETS:
        if p.width == width and p.height == height:
            return p
    return None


# Nomi ComfyUI: sampler_name e scheduler sono campi separati nel KSampler.
SAMPLERS: tuple[str, ...] = (
    "euler",
    "euler_ancestral",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "dpmpp_sde",
    "dpmpp_2s_ancestral",
    "ddim",
    "uni_pc",
)

SCHEDULERS: tuple[str, ...] = (
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
)

UPSCALE_FACTORS: tuple[float, ...] = (1.0, 1.5, 2.0, 4.0)

DEFAULT_SAMPLER = "dpmpp_2m"
DEFAULT_SCHEDULER = "karras"


# ---------------------------------------------------------------------------
# Costruzione parametri per il RecipeWorker + stima costo
# ---------------------------------------------------------------------------


@dataclass
class GenerationForm:
    """Stato grezzo del form di generazione (valori primitivi, no widget)."""

    prompt: str = ""
    negative: str = ""
    width: int = 832
    height: int = 1216
    steps: int = 30
    cfg: float = 7.0
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
    seed: int = -1
    lora_weight: float = 0.85
    batch: int = 1
    upscale_factor: float = 1.0
    use_controlnet: bool = False
    use_ipadapter: bool = False
    # LoRA extra selezionati dall'utente (oltre a quello del progetto).
    # extra_lora_count guida la stima costo; extra_loras porta i path/peso reali
    # passati al worker per lo stacking. La UI tiene i due allineati.
    extra_lora_count: int = 0
    extra_loras: tuple[tuple[str, float], ...] = ()

    def to_user_params(self) -> dict:
        """Dict passato a RecipeWorker. Chiavi non mappate dal workflow vengono
        ignorate silenziosamente dal worker, quindi è sicuro includerle tutte."""
        return {
            "prompt": self.prompt,
            "negative": self.negative,
            "width": int(self.width),
            "height": int(self.height),
            "steps": int(self.steps),
            "cfg": float(self.cfg),
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "seed": int(self.seed),
            "lora_weight": float(self.lora_weight),
            "batch": int(self.batch),
            "upscale_factor": float(self.upscale_factor),
            "loras": [
                {"path": path, "weight": float(weight)}
                for path, weight in self.extra_loras
            ],
        }

    def to_generation_config(self) -> GenerationConfig:
        """Costruisce un GenerationConfig (usato per la stima costo)."""
        return GenerationConfig(
            prompt=self.prompt,
            negative_prompt=self.negative,
            width=int(self.width),
            height=int(self.height),
            steps=int(self.steps),
            cfg_scale=float(self.cfg),
            sampler=self.sampler,
            seed=int(self.seed),
            batch_size=int(self.batch),
            loras=[LoraSpec(path=f"slot{i}") for i in range(self.extra_lora_count)],
        )

    def estimate(self, cost_model: Optional[CostModel] = None) -> CostBreakdown:
        """Stima il costo in crediti di questa configurazione."""
        return estimate_cost(
            self.to_generation_config(),
            cost_model,
            upscale_factor=self.upscale_factor,
            use_controlnet=self.use_controlnet,
            use_ipadapter=self.use_ipadapter,
        )
