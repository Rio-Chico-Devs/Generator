"""Definizione delle ricette (pipeline composite).

Ogni ricetta mappa input utente → parametri di un workflow ComfyUI.
La logica di esecuzione vera sta in workers/recipe_worker.py; qui si
definisce COSA serve e COME si parametrizza.

Riferimento: docs/RECIPES.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class RecipeId(str, Enum):
    BASE = "base_style"
    CHARACTER_IN_POSE = "character_in_pose"
    PRODUCT_IN_SCENE = "product_in_scene"
    CORRECT = "correct_inpaint"
    REFINE = "hires_refine"


@dataclass
class RecipeInput:
    """Descrizione di un input richiesto da una ricetta (per generare UI)."""

    key: str
    label: str
    kind: str  # "text", "image", "pose", "character", "mask", "choice", "number"
    required: bool = True
    default: object = None
    choices: Optional[list[str]] = None
    advanced: bool = False  # se True, sotto pannello "Avanzate"
    help: str = ""


@dataclass
class RecipeDef:
    id: RecipeId
    name: str
    description: str
    workflow_file: str  # nome file in assets/workflows/
    inputs: list[RecipeInput]
    required_models: list[str]  # id da catalog, per check pre-esecuzione
    priority_phase: int  # in quale fase roadmap viene implementata


# Definizioni delle ricette ----------------------------------------------

RECIPES: dict[RecipeId, RecipeDef] = {
    RecipeId.BASE: RecipeDef(
        id=RecipeId.BASE,
        name="Genera nel mio stile",
        description="Prompt testuale → immagine nel tuo stile appreso.",
        workflow_file="base_txt2img.json",
        required_models=["pony-v6-xl"],
        priority_phase=1,
        inputs=[
            RecipeInput("prompt", "Descrizione", "text"),
            RecipeInput("negative", "Da evitare", "text", required=False,
                        default="low quality, worst quality, blurry"),
            RecipeInput("width", "Larghezza", "number", default=1024, advanced=True),
            RecipeInput("height", "Altezza", "number", default=1024, advanced=True),
            RecipeInput("steps", "Step", "number", default=30, advanced=True),
            RecipeInput("cfg", "CFG", "number", default=7.0, advanced=True),
            RecipeInput("seed", "Seed", "number", default=-1, advanced=True,
                        help="-1 = casuale"),
            RecipeInput("lora_weight", "Forza stile", "number", default=0.85,
                        advanced=True),
        ],
    ),
    RecipeId.CHARACTER_IN_POSE: RecipeDef(
        id=RecipeId.CHARACTER_IN_POSE,
        name="Posa da foto",
        description=(
            "Ridisegna il tuo personaggio (LoRA attiva) nella stessa posa di "
            "una foto di riferimento, nel tuo stile — accessori e identità "
            "restano quelli del personaggio, cambia solo la posizione."
        ),
        workflow_file="character_in_pose.json",
        required_models=[
            "pony-v6-xl",
            "controlnet-openpose-sdxl",
            "controlnet-depth-sdxl",
        ],
        priority_phase=1,
        inputs=[
            RecipeInput("pose", "Foto di riferimento", "pose",
                        help="Una foto o disegno nella posa che vuoi replicare"),
            RecipeInput("prompt", "Descrizione", "text"),
            RecipeInput("negative", "Da evitare", "text", required=False,
                        default="low quality, worst quality, bad hands, deformed"),
            RecipeInput("pose_weight", "Fedeltà posa", "number", default=0.5,
                        advanced=True,
                        help="Bassa = più libertà anatomica, alta = segue lo scheletro alla lettera"),
            RecipeInput("depth_weight", "Coerenza spaziale", "number",
                        default=0.3, advanced=True,
                        help="Rinforza le proporzioni/profondità senza irrigidire la posa"),
            RecipeInput("lora_weight", "Forza stile", "number", default=0.85,
                        advanced=True),
            RecipeInput("autofix_hands", "Correggi mani", "choice",
                        default="yes", choices=["yes", "no"], advanced=True),
            RecipeInput("autofix_face", "Correggi volto", "choice",
                        default="yes", choices=["yes", "no"], advanced=True),
            RecipeInput("seed", "Seed", "number", default=-1, advanced=True),
        ],
    ),
    RecipeId.PRODUCT_IN_SCENE: RecipeDef(
        id=RecipeId.PRODUCT_IN_SCENE,
        name="Prodotto in scena",
        description=(
            "Crea materiale promozionale attorno a un prodotto reale, "
            "nel tuo stile, con il prodotto integrato coerentemente."
        ),
        workflow_file="product_in_scene.json",
        required_models=["pony-v6-xl", "birefnet", "ic-light"],
        priority_phase=4,
        inputs=[
            RecipeInput("product", "Prodotto", "image",
                        help="Foto del prodotto (dalla pool o caricata)"),
            RecipeInput("scene_prompt", "Scena desiderata", "text"),
            RecipeInput("template", "Formato", "choice", default="banner",
                        choices=["volantino_a4", "post_quadrato", "story_verticale",
                                 "banner", "og_image"]),
            RecipeInput("title_text", "Testo titolo", "text", required=False),
            RecipeInput("subtitle_text", "Sottotitolo", "text", required=False),
            RecipeInput("lora_weight", "Forza stile", "number", default=0.8,
                        advanced=True),
            RecipeInput("seed", "Seed", "number", default=-1, advanced=True),
        ],
    ),
    RecipeId.REFINE: RecipeDef(
        id=RecipeId.REFINE,
        name="Migliora (Hires)",
        description=(
            "Prende un'immagine già generata, la ingrandisce e fa un secondo "
            "passaggio leggero che aggiunge dettaglio e nitidezza senza "
            "stravolgere la composizione."
        ),
        workflow_file="hires_refine.json",
        required_models=[],
        priority_phase=1,
        inputs=[
            RecipeInput("image", "Immagine", "image"),
            RecipeInput("prompt", "Descrizione", "text", required=False),
            RecipeInput("negative", "Da evitare", "text", required=False,
                        default="low quality, worst quality, blurry"),
            RecipeInput("scale_by", "Ingrandimento", "number", default=1.5,
                        advanced=True, help="1.0 = stessa dimensione, 2.0 = doppia"),
            RecipeInput("denoise", "Intensità", "number", default=0.4,
                        advanced=True,
                        help="Bassa = ritocca, alta = ridisegna (0.3–0.5 consigliato)"),
            RecipeInput("lora_weight", "Forza stile", "number", default=0.85,
                        advanced=True),
            RecipeInput("seed", "Seed", "number", default=-1, advanced=True,
                        help="-1 = casuale"),
        ],
    ),
    RecipeId.CORRECT: RecipeDef(
        id=RecipeId.CORRECT,
        name="Correggi / rifinisci",
        description=(
            "Sistema una zona di un'immagine senza toccare il resto e "
            "senza aggiungere elementi non richiesti."
        ),
        workflow_file="correct_inpaint.json",
        required_models=["pony-v6-xl"],
        priority_phase=5,
        inputs=[
            RecipeInput("image", "Immagine", "image"),
            RecipeInput("mask", "Zona da correggere", "mask",
                        help="Disegna sull'area da sistemare"),
            RecipeInput("prompt", "Cosa correggere", "text", required=False),
            RecipeInput("strength", "Intensità", "number", default=0.5,
                        help="Bassa = corregge, alta = ridisegna", advanced=True),
            RecipeInput("variants", "Varianti", "number", default=4,
                        advanced=True),
            RecipeInput("seed", "Seed", "number", default=-1, advanced=True),
        ],
    ),
}


def get_recipe(recipe_id: RecipeId) -> RecipeDef:
    return RECIPES[recipe_id]


def recipes_for_phase(phase: int) -> list[RecipeDef]:
    """Ricette implementate fino a una certa fase (per gating UI)."""
    return [r for r in RECIPES.values() if r.priority_phase <= phase]
