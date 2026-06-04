"""Gestione Pose Library.

Le ~500 reference di pose: import, categorizzazione, ricerca, cache
scheletri. NON è un dataset di training, è un catalogo di input
ControlNet.

Riferimento: docs/POSE_LIBRARY.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.atomic import atomic_write_text
from src.utils.paths import get_user_data_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass
class PoseEntry:
    id: str
    source_path: str
    skeleton_path: str = ""
    skeleton_preview: str = ""
    thumbnail: str = ""
    added_at: str = ""
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    person_count: int = 1
    confidence: float = 0.0
    user_notes: str = ""
    favorite: bool = False


class PoseLibrary:
    """Catalogo pose su filesystem, indicizzato in memoria."""

    def __init__(self) -> None:
        self.root = get_user_data_dir() / "pose_library"
        self._poses: dict[str, PoseEntry] = {}
        self._loaded = False

    # --- Path -----------------------------------------------------

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def skeletons_dir(self) -> Path:
        return self.root / "skeletons"

    @property
    def thumbnails_dir(self) -> Path:
        return self.root / "thumbnails"

    @property
    def index_path(self) -> Path:
        return self.root / "library.json"

    # --- Load / save ----------------------------------------------

    def ensure_dirs(self) -> None:
        for d in (self.sources_dir, self.skeletons_dir, self.thumbnails_dir):
            d.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        self.ensure_dirs()
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._poses = {
                pid: PoseEntry(id=pid, **{k: v for k, v in pdata.items() if k != "id"})
                for pid, pdata in data.get("poses", {}).items()
            }
        self._loaded = True
        logger.info("Pose library caricata: %d pose", len(self._poses))

    def save(self) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "poses": {pid: _entry_dict(e) for pid, e in self._poses.items()},
        }
        atomic_write_text(
            self.index_path, json.dumps(data, indent=2, ensure_ascii=False)
        )

    # --- Query ----------------------------------------------------

    def all(self) -> list[PoseEntry]:
        return list(self._poses.values())

    def get(self, pose_id: str) -> Optional[PoseEntry]:
        return self._poses.get(pose_id)

    def search(
        self,
        text: str = "",
        categories: Optional[list[str]] = None,
        favorites_only: bool = False,
    ) -> list[PoseEntry]:
        """Filtro in memoria su metadata."""
        results = []
        text_low = text.lower()
        for e in self._poses.values():
            if favorites_only and not e.favorite:
                continue
            if categories and not all(c in e.categories for c in categories):
                continue
            if text_low:
                haystack = " ".join([*e.tags, *e.categories, e.user_notes]).lower()
                if text_low not in haystack:
                    continue
            results.append(e)
        return results

    def all_categories(self) -> list[str]:
        cats: set[str] = set()
        for e in self._poses.values():
            cats.update(e.categories)
        return sorted(cats)

    # --- Mutations ------------------------------------------------

    def add(self, entry: PoseEntry) -> None:
        self._poses[entry.id] = entry

    def update(self, entry: PoseEntry) -> None:
        self._poses[entry.id] = entry

    def remove(self, pose_id: str) -> None:
        self._poses.pop(pose_id, None)

    def next_id(self) -> str:
        n = len(self._poses) + 1
        while f"pose_{n:04d}" in self._poses:
            n += 1
        return f"pose_{n:04d}"


# --- Joint parsing helpers -------------------------------------------

# OpenPose-18 integer index → canonical name
_OP18: list[str] = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist", "mid_hip",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

# DWPose / COCO-17 string name → canonical name
_COCO17: dict[str, str] = {
    "nose": "nose",
    "left_eye": "l_eye", "right_eye": "r_eye",
    "left_ear": "l_ear", "right_ear": "r_ear",
    "left_shoulder": "l_shoulder", "right_shoulder": "r_shoulder",
    "left_elbow": "l_elbow", "right_elbow": "r_elbow",
    "left_wrist": "l_wrist", "right_wrist": "r_wrist",
    "left_hip": "l_hip", "right_hip": "r_hip",
    "left_knee": "l_knee", "right_knee": "r_knee",
    "left_ankle": "l_ankle", "right_ankle": "r_ankle",
    # ShortHand aliases (OpenPose string names)
    "neck": "neck", "mid_hip": "mid_hip",
    "r_shoulder": "r_shoulder", "l_shoulder": "l_shoulder",
    "r_elbow": "r_elbow", "l_elbow": "l_elbow",
    "r_wrist": "r_wrist", "l_wrist": "l_wrist",
    "r_hip": "r_hip", "l_hip": "l_hip",
    "r_knee": "r_knee", "l_knee": "l_knee",
    "r_ankle": "r_ankle", "l_ankle": "l_ankle",
    "r_eye": "r_eye", "l_eye": "l_eye",
    "r_ear": "r_ear", "l_ear": "l_ear",
}

_CONF_MIN = 0.3   # joint ignorato se confidence < 0.3


def _parse_joints(joints: dict) -> dict[str, tuple[float, float]]:
    """Converte il dict di joint grezzi in {nome_canonico: (x, y)}.

    Supporta:
    - chiavi intere (OpenPose-18 index)
    - chiavi stringa COCO-17 o shorthand
    - valori [x, y, conf], (x, y, conf) o {"x": .., "y": .., "conf": ..}
    Esclude joint con confidence < _CONF_MIN.
    Coordinate normalizzate [0, 1], origine top-left.
    """
    result: dict[str, tuple[float, float]] = {}
    for key, val in joints.items():
        if isinstance(key, int):
            canon = _OP18[key] if 0 <= key < len(_OP18) else None
        else:
            canon = _COCO17.get(str(key).lower()) or _COCO17.get(str(key))
        if canon is None:
            continue

        if isinstance(val, (list, tuple)) and len(val) >= 3:
            x, y, conf = float(val[0]), float(val[1]), float(val[2])
        elif isinstance(val, (list, tuple)) and len(val) == 2:
            x, y, conf = float(val[0]), float(val[1]), 1.0
        elif isinstance(val, dict):
            x = float(val.get("x", 0.0))
            y = float(val.get("y", 0.0))
            conf = float(val.get("conf", val.get("confidence", val.get("score", 1.0))))
        else:
            continue

        if conf >= _CONF_MIN:
            result[canon] = (x, y)
    return result


def _avg_y(j: dict[str, tuple[float, float]], *keys: str) -> float | None:
    vals = [j[k][1] for k in keys if k in j]
    return sum(vals) / len(vals) if vals else None


def categorize_skeleton(joints: dict) -> list[str]:
    """Categorizza automaticamente una posa dalle posizioni dei joint DWPose.

    Parametro joints: dict con chiavi intere (OpenPose-18) o stringhe
    (COCO-17 / shorthand) e valori [x, y, confidence] con coordinate
    normalizzate in [0, 1] (origine top-left, y cresce verso il basso).

    Restituisce una lista di categorie come
    ["standing", "frontal", "arms_raised"].
    """
    j = _parse_joints(joints)
    if not j:
        return []

    cats: list[str] = []

    shoulder_y = _avg_y(j, "l_shoulder", "r_shoulder")
    hip_y      = _avg_y(j, "l_hip", "r_hip", "mid_hip")
    knee_y     = _avg_y(j, "l_knee", "r_knee")
    ankle_y    = _avg_y(j, "l_ankle", "r_ankle")

    # --- Postura (standing / sitting / lying) ---
    if shoulder_y is not None and hip_y is not None:
        all_x = [v[0] for v in j.values()]
        all_y = [v[1] for v in j.values()]
        horiz_span = max(all_x) - min(all_x) if all_x else 0.0
        total_vert = max(all_y) - min(all_y) if all_y else 0.0

        if total_vert < 0.2 and horiz_span > 0.4:
            # Corpo quasi orizzontale → sdraiato
            cats.append("lying")
        elif (ankle_y is not None and shoulder_y < hip_y < ankle_y
              and (ankle_y - shoulder_y) > 0.35):
            cats.append("standing")
        elif knee_y is not None and abs(knee_y - hip_y) < 0.15:
            # Ginocchia alla stessa altezza dei fianchi → seduto/accucciato
            cats.append("sitting")
        elif shoulder_y < hip_y:
            cats.append("standing")  # gerarchia verticale sufficiente
        else:
            cats.append("standing")  # default

    # --- Braccia alzate ---
    wrist_y = _avg_y(j, "l_wrist", "r_wrist")
    if wrist_y is not None and shoulder_y is not None:
        if wrist_y < shoulder_y:   # y cresce verso il basso: polso sopra spalla
            cats.append("arms_raised")

    # --- Orientamento (frontal / profile / posterior) ---
    has_nose  = "nose"  in j
    has_l_eye = "l_eye" in j
    has_r_eye = "r_eye" in j
    has_l_ear = "l_ear" in j
    has_r_ear = "r_ear" in j

    if has_nose and has_l_eye and has_r_eye:
        cats.append("frontal")
    elif (has_l_eye) != (has_r_eye) or (has_l_ear) != (has_r_ear):
        # Un solo occhio o un solo orecchio visibile → profilo
        cats.append("profile")
    elif not has_nose and not has_l_eye and not has_r_eye:
        if "l_shoulder" in j or "r_shoulder" in j:
            cats.append("posterior")

    return cats


def _entry_dict(e: PoseEntry) -> dict:
    d = asdict(e)
    d.pop("id", None)
    return d


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
