"""Preset di prompting per dialetto di modello.

Modelli diversi rispondono a "dialetti" di prompt diversi:

- **Pony** vuole i tag ``score_9, score_8_up...`` in testa al positivo;
- gli SDXL anime (**Illustrious**, Animagine) usano tag di qualità booru
  (``masterpiece, best quality, very aesthetic...``) e CLIP skip 2;
- i modelli **fotografici** preferiscono prompt naturali senza quality-tag.

Questi preset incapsulano i quality-tag, i negative di qualità e il CLIP skip
giusti per ogni dialetto, così la resa si avvicina a quella di piattaforme
come TensorArt senza che l'utente debba ricordare la sintassi di ogni modello.

Modulo puro: nessuna dipendenza da Qt, ComfyUI o catalogo. Testabile da solo.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityPreset:
    """Tag di qualità e CLIP skip per un dialetto di modello."""

    dialect: str
    positive_prefix: str   # tag di qualità anteposti al prompt utente
    negative_suffix: str   # tag di qualità accodati al negative
    clip_skip: int         # 1 = nessuno skip; 2 = salta l'ultimo layer CLIP
    # Nota: in ComfyUI il nodo CLIPSetLastLayer usa il valore NEGATIVO
    # (stop_at_clip_layer = -clip_skip). La conversione avviene nel worker.


PRESETS: dict[str, QualityPreset] = {
    "pony": QualityPreset(
        dialect="pony",
        positive_prefix="score_9, score_8_up, score_7_up, score_6_up,",
        negative_suffix=(
            "score_6, score_5, score_4, worst quality, low quality, lowres, "
            "bad anatomy, bad hands, missing fingers, extra digits, watermark, "
            "signature,"
        ),
        clip_skip=2,
    ),
    "illustrious": QualityPreset(
        dialect="illustrious",
        positive_prefix="masterpiece, best quality, very aesthetic, absurdres, newest,",
        negative_suffix=(
            "worst quality, low quality, lowres, bad anatomy, bad hands, "
            "missing fingers, extra digits, jpeg artifacts, signature, "
            "watermark, blurry,"
        ),
        clip_skip=2,
    ),
    "anime": QualityPreset(
        dialect="anime",
        positive_prefix="masterpiece, best quality, highres,",
        negative_suffix=(
            "worst quality, low quality, lowres, bad anatomy, bad hands, "
            "jpeg artifacts, signature, watermark,"
        ),
        clip_skip=2,
    ),
    "photo": QualityPreset(
        dialect="photo",
        positive_prefix="",
        negative_suffix="low quality, worst quality, blurry, jpeg artifacts, deformed,",
        clip_skip=1,
    ),
    "plain": QualityPreset(
        dialect="plain",
        positive_prefix="",
        negative_suffix="",
        clip_skip=1,
    ),
}

DEFAULT_DIALECT = "plain"


def get_preset(dialect: str | None) -> QualityPreset:
    """Preset per il dialetto dato; ``plain`` se ignoto o assente."""
    return PRESETS.get(dialect or DEFAULT_DIALECT, PRESETS[DEFAULT_DIALECT])


def merge_tags(*chunks: str) -> str:
    """Unisce spezzoni di prompt separati da virgola, deduplicando i tag.

    Confronto case-insensitive, ordine di prima apparizione preservato.
    Evita che i quality-tag del preset si duplichino con quelli già presenti
    nel trigger del progetto o nel prompt dell'utente.
    """
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        for raw in chunk.split(","):
            tag = raw.strip()
            if not tag:
                continue
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tag)
    return ", ".join(out)


@dataclass(frozen=True)
class BuiltPrompt:
    positive: str
    negative: str
    clip_skip: int


def build_prompts(
    preset: QualityPreset,
    *,
    user_positive: str,
    trigger_prefix: str = "",
    user_negative: str = "",
    project_negative: str = "",
) -> BuiltPrompt:
    """Compone positivo e negativo finali applicando il preset del dialetto.

    Positivo: ``preset.positive_prefix`` → ``trigger_prefix`` → ``user_positive``
    (i quality-tag del modello vanno in testa, poi l'attivatore del personaggio,
    poi la richiesta dell'utente).

    Negativo: il negativo dell'utente (o, se assente, quello di progetto)
    seguito dai negative di qualità del preset. Tutto deduplicato.
    """
    positive = merge_tags(preset.positive_prefix, trigger_prefix, user_positive)
    base_negative = user_negative.strip() if user_negative.strip() else project_negative
    negative = merge_tags(base_negative, preset.negative_suffix)
    return BuiltPrompt(positive=positive, negative=negative, clip_skip=preset.clip_skip)
