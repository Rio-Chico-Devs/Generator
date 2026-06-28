"""Catalogo modelli: base (CATALOG) + ausiliari (AUX_CATALOG) + VAE.

I modelli base sono i checkpoint SDXL/SD1.5 usati per la generazione.
I modelli ausiliari sono i componenti specializzati che le ricette usano
in pipeline (ControlNet, IP-Adapter, segmentazione, relighting, upscale).

Riferimento: docs/MODELS.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ModelFamily = Literal["sdxl", "sd15", "flux"]
UseCase = Literal["character", "photo", "illustration", "abstract", "logo"]
# Dialetto di prompting: determina quality-tag e CLIP skip (vedi core/prompting.py).
# Indipendente dalla famiglia architetturale: Pony e Illustrious sono entrambi
# SDXL ma vogliono prompt diversi.
PromptDialect = Literal["pony", "illustrious", "anime", "photo", "plain"]


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    family: ModelFamily
    repo_id: str
    revision: Optional[str]
    size_gb: float
    license: str
    license_url: str
    suitable_for: tuple[UseCase, ...]
    requires_vae: Optional[str] = None
    min_vram_gb_inference: float = 5.0
    min_vram_gb_training: float = 7.0
    commercial_use_ok: bool = True
    description: str = ""
    prompt_dialect: PromptDialect = "plain"


CATALOG: dict[str, ModelEntry] = {
    # --- Checkpoint personali di Bru (scaricati da Tensor.art, non da HF) ---
    # In cima di proposito: così sono i primi nei menu e un progetto creato
    # senza scegliere esplicitamente usa un modello che l'utente possiede
    # davvero (non un default del catalogo non scaricato).
    # id = nome file esatto (senza estensione): RecipeWorker costruisce il nome
    # del checkpoint per ComfyUI come f"{model_id}.safetensors", quindi deve
    # combaciare col file reale in models/checkpoints. repo_id/license sono solo
    # informativi: questi modelli non passano dal download manager.
    "WAI-Nsfw-Illustrious-17": ModelEntry(
        id="WAI-Nsfw-Illustrious-17",
        name="WAI NSFW Illustrious v17",
        family="sdxl",
        repo_id="tensor.art (locale, non scaricato da HF)",
        revision=None,
        size_gb=6.46,
        license="Da verificare (CivitAI/Tensor.art, termini originali non confermati)",
        license_url="",
        suitable_for=("character", "illustration"),
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=False,
        description="Checkpoint personale, basato su Illustrious. Stessa convenzione di prompt di illustrious-xl.",
        prompt_dialect="illustrious",
    ),
    "reedNSFWIllustrious_v40": ModelEntry(
        id="reedNSFWIllustrious_v40",
        name="ReedNSFW Illustrious v40",
        family="sdxl",
        repo_id="tensor.art (locale, non scaricato da HF)",
        revision=None,
        size_gb=6.62,
        license="Da verificare (CivitAI/Tensor.art, termini originali non confermati)",
        license_url="",
        suitable_for=("character", "illustration"),
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=False,
        description="Checkpoint personale, basato su Illustrious. Stessa convenzione di prompt di illustrious-xl.",
        prompt_dialect="illustrious",
    ),
    "pony-v6-xl": ModelEntry(
        id="pony-v6-xl",
        name="Pony Diffusion V6 XL",
        family="sdxl",
        repo_id="AstraliteHeart/pony-diffusion-v6-xl",
        revision=None,
        size_gb=7.0,
        license="Fair AI Public 1.0-SD",
        license_url="https://freedevproject.org/faipl-1.0-sd/",
        suitable_for=("character", "illustration"),
        requires_vae="sdxl-vae-fp16-fix",
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=True,
        description=(
            "Modello SDXL specializzato in character art e illustrazione, "
            "ottimizzato per tag Danbooru. Ottimo per personaggi originali, "
            "concept art, sprite stilizzati."
        ),
        prompt_dialect="pony",
    ),
    "illustrious-xl": ModelEntry(
        id="illustrious-xl",
        name="Illustrious XL",
        family="sdxl",
        repo_id="OnomaAIResearch/Illustrious-xl-early-release-v0",
        revision=None,
        size_gb=6.9,
        license="Fair AI Public 1.0-SD",
        license_url="https://freedevproject.org/faipl-1.0-sd/",
        suitable_for=("character", "illustration"),
        requires_vae="sdxl-vae-fp16-fix",
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=True,
        description=(
            "Modello SDXL anime/illustrazione ad alta fedeltà di tag, base di "
            "moltissimi fine-tune della community. Vuole i quality-tag booru "
            "(masterpiece, best quality, very aesthetic...) e CLIP skip 2."
        ),
        prompt_dialect="illustrious",
    ),
    "juggernaut-xl-v9": ModelEntry(
        id="juggernaut-xl-v9",
        name="Juggernaut XL v9",
        family="sdxl",
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        revision=None,
        size_gb=7.0,
        license="OpenRAIL++-M",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
        suitable_for=("photo", "illustration"),
        requires_vae="sdxl-vae-fp16-fix",
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=True,
        description=(
            "Fine-tune SDXL ottimizzato per fotorealismo. "
            "Ottimo per render prodotti, mockup, asset realistici."
        ),
        prompt_dialect="photo",
    ),
    "animagine-xl-4": ModelEntry(
        id="animagine-xl-4",
        name="Animagine XL 4.0",
        family="sdxl",
        repo_id="cagliostrolab/animagine-xl-4.0",
        revision=None,
        size_gb=7.0,
        license="Fair AI Public 1.0-SD",
        license_url="https://freedevproject.org/faipl-1.0-sd/",
        suitable_for=("character", "illustration"),
        requires_vae="sdxl-vae-fp16-fix",
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=True,
        description=(
            "Modello SDXL per illustrazione anime/manga alta qualità. "
            "Alternativa a Pony con output più 'pulito'."
        ),
        prompt_dialect="anime",
    ),
    "realistic-vision-v6": ModelEntry(
        id="realistic-vision-v6",
        name="Realistic Vision V6 (SD 1.5)",
        family="sd15",
        repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE",
        revision=None,
        size_gb=4.0,
        license="CreativeML OpenRAIL-M",
        license_url="https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        suitable_for=("photo", "illustration"),
        min_vram_gb_inference=4.0,
        min_vram_gb_training=5.5,
        commercial_use_ok=True,
        description=(
            "Modello SD 1.5 leggero per fotorealismo veloce. "
            "Training in 30-45 min, ideale per esperimenti."
        ),
        prompt_dialect="photo",
    ),
    "sd-1-5": ModelEntry(
        id="sd-1-5",
        name="Stable Diffusion 1.5",
        family="sd15",
        repo_id="runwayml/stable-diffusion-v1-5",
        revision=None,
        size_gb=4.0,
        license="CreativeML OpenRAIL-M",
        license_url="https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        suitable_for=("character", "illustration", "abstract"),
        min_vram_gb_inference=4.0,
        min_vram_gb_training=5.5,
        commercial_use_ok=True,
        description=(
            "Modello SD 1.5 base. Versatile, leggero, training rapido. "
            "Sweet spot per sprite, pixel art, esperimenti."
        ),
    ),
}


# VAE separati (sostituiscono i VAE buggy dei modelli base)
@dataclass(frozen=True)
class VaeEntry:
    id: str
    repo_id: str
    filename: str
    size_mb: float
    license: str


VAE_CATALOG: dict[str, VaeEntry] = {
    "sdxl-vae-fp16-fix": VaeEntry(
        id="sdxl-vae-fp16-fix",
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename="sdxl_vae.safetensors",
        size_mb=335.0,
        license="MIT",
    ),
}


# ---------------------------------------------------------------------------
# Modelli ausiliari — componenti delle ricette
# ---------------------------------------------------------------------------

AuxKind = Literal[
    "controlnet",   # guida composizione/posa (ControlNet)
    "ipadapter",    # trasferimento identità/stile (IP-Adapter)
    "detector",     # estrazione feature (posa, segmentazione)
    "segmentation", # rimozione sfondo (BiRefNet, RMBG)
    "relighting",   # re-illuminazione prodotti (IC-Light)
    "upscaler",     # upscaling output
    "adetailer",    # auto-fix visi/mani (YOLO detection)
]


@dataclass(frozen=True)
class AuxModelEntry:
    """Un modello ausiliare usato in pipeline dalle ricette.

    A differenza di ModelEntry (checkpoint base), questi modelli sono
    file singoli (.safetensors / .bin / .pth) caricati on-demand da
    ComfyUI nei nodi specializzati.
    """

    id: str              # deve corrispondere a required_models in RecipeDef
    name: str
    kind: AuxKind
    repo_id: str         # HuggingFace repo (owner/name)
    filename: str        # file specifico da scaricare dal repo
    comfy_subdir: str    # sotto-cartella in ComfyUI/models/ (per download manager)
    size_gb: float
    license: str
    license_url: str
    description: str
    min_vram_gb: float = 0.5   # VRAM aggiuntiva quando il nodo è attivo
    required_for_phase: int = 1  # prima fase che richiede questo modello
    commercial_use_ok: bool = True


AUX_CATALOG: dict[str, AuxModelEntry] = {
    # --- Ricetta A: Personaggio in posa (Fase 3) ---

    "controlnet-openpose-sdxl": AuxModelEntry(
        id="controlnet-openpose-sdxl",
        name="ControlNet OpenPose SDXL",
        kind="controlnet",
        repo_id="thibaud/controlnet-openpose-sdxl-1.0",
        filename="controlnet-pose-sdxl-1.0.safetensors",
        comfy_subdir="controlnet",
        size_gb=2.5,
        license="OpenRAIL",
        license_url="https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/blob/main/LICENSE",
        description=(
            "Guida la posa del personaggio usando lo scheletro estratto da DWPose. "
            "Input: mappa scheletro (keypoints). Usato in Ricetta A."
        ),
        min_vram_gb=2.5,
        required_for_phase=3,
        commercial_use_ok=True,
    ),

    "ipadapter-faceid-plus-v2": AuxModelEntry(
        id="ipadapter-faceid-plus-v2",
        name="IP-Adapter FaceID Plus v2 SDXL",
        kind="ipadapter",
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid-plusv2_sdxl.bin",
        comfy_subdir="ipadapter",
        size_gb=1.0,
        license="Apache-2.0",
        license_url="https://huggingface.co/h94/IP-Adapter-FaceID/blob/main/LICENSE",
        description=(
            "Mantiene l'identità visiva del personaggio (volto + caratteristiche) "
            "attraverso i frame generati. Input: 1-3 immagini reference del personaggio. "
            "Usato in Ricetta A."
        ),
        min_vram_gb=1.0,
        required_for_phase=3,
        commercial_use_ok=True,
    ),

    "dwpose": AuxModelEntry(
        id="dwpose",
        name="DWPose (body + hand detector)",
        kind="detector",
        repo_id="yzd-v/DWPose",
        filename="dw-ll_ucoco_384.onnx",  # modello pose; vedi anche det_model
        comfy_subdir="onnx",
        size_gb=0.4,
        license="Apache-2.0",
        license_url="https://huggingface.co/yzd-v/DWPose/blob/main/LICENSE",
        description=(
            "Estrae lo scheletro (keypoints corpo + mani) da un'immagine reference. "
            "Output usato come input per controlnet-openpose-sdxl. "
            "Gestito da comfyui_controlnet_aux; il download avviene automaticamente "
            "al primo utilizzo del nodo DWPose Estimator in ComfyUI."
        ),
        min_vram_gb=0.2,
        required_for_phase=1,  # serve per Pose Library (estrazione scheletro)
        commercial_use_ok=True,
    ),

    # --- Ricetta B: Prodotto in scena (Fase 4) ---

    "birefnet": AuxModelEntry(
        id="birefnet",
        name="BiRefNet (background removal)",
        kind="segmentation",
        repo_id="ZhengPeng7/BiRefNet",
        filename="BiRefNet-general-epoch_244.pth",
        comfy_subdir="BiRefNet",
        size_gb=0.9,
        license="MIT",
        license_url="https://huggingface.co/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        description=(
            "Rimozione sfondo ad alta precisione: isola il prodotto dalla foto "
            "originale mantenendo bordi nitidi (capelli, trasparenze). "
            "Sostituto locale di remove.bg. Usato in Ricetta B."
        ),
        min_vram_gb=0.5,
        required_for_phase=4,
        commercial_use_ok=True,
    ),

    "ic-light": AuxModelEntry(
        id="ic-light",
        name="IC-Light (relighting)",
        kind="relighting",
        repo_id="lllyasviel/ic-light",
        filename="iclight_sd15_fc.safetensors",
        comfy_subdir="unet",
        size_gb=1.7,
        license="Apache-2.0",
        license_url="https://huggingface.co/lllyasviel/ic-light/blob/main/LICENSE",
        description=(
            "Re-illumina il prodotto per integrarlo coerentemente nella scena "
            "generata (luce, ombra, colore ambientale). Modello SD 1.5 — usato "
            "per il solo step di relighting, non per la generazione SDXL principale. "
            "Usato in Ricetta B."
        ),
        min_vram_gb=1.5,
        required_for_phase=4,
        commercial_use_ok=True,
    ),

    # --- Upscale — tutte le ricette (Fase 1+) ---

    "real-esrgan-x4plus": AuxModelEntry(
        id="real-esrgan-x4plus",
        name="Real-ESRGAN x4plus",
        kind="upscaler",
        repo_id="ai-forever/Real-ESRGAN",
        filename="RealESRGAN_x4plus.pth",
        comfy_subdir="upscale_models",
        size_gb=0.07,
        license="BSD-3-Clause",
        license_url="https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE",
        description=(
            "Upscaler generico 4x per output finali. Ottimo su illustrazioni "
            "e character art. Usato opzionalmente in tutte le ricette."
        ),
        min_vram_gb=0.1,
        required_for_phase=1,
        commercial_use_ok=True,
    ),
}


def aux_models_for_phase(phase: int) -> list[AuxModelEntry]:
    """Modelli ausiliari necessari fino alla fase indicata."""
    return [m for m in AUX_CATALOG.values() if m.required_for_phase <= phase]


def dialect_for_model(model_id: str) -> str:
    """Dialetto di prompting per un id modello; ``plain`` se sconosciuto."""
    entry = CATALOG.get(model_id)
    return entry.prompt_dialect if entry else "plain"


# ---------------------------------------------------------------------------
# Helper per suggerire modello in base al caso d'uso
# ---------------------------------------------------------------------------
def suggest_models_for(use_case: UseCase, prefer_fast: bool = False) -> list[str]:
    """Restituisce ID modelli ordinati per adeguatezza."""
    matches = [
        (mid, m) for mid, m in CATALOG.items() if use_case in m.suitable_for
    ]
    # Ordina: SDXL prima se prefer_fast=False, SD 1.5 prima se True
    matches.sort(
        key=lambda x: (
            x[1].family == "sd15" if not prefer_fast else x[1].family != "sd15",
            x[1].size_gb,
        )
    )
    return [mid for mid, _ in matches]
