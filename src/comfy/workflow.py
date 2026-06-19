"""Caricamento e parametrizzazione dei workflow ComfyUI.

I file in ``assets/workflows/*.json`` sono template ComfyUI in formato
API (un dizionario ``node_id -> {"class_type", "inputs"}``). Un file di
mapping affiancato ``<nome>.map.json`` dichiara i *ruoli semantici*
(seed, prompt, lora, dimensioni...) e a quale nodo/campo corrispondono,
così il codice dell'app non deve hardcodare gli id dei nodi: se il
workflow cambia, basta aggiornare il mapping.

Riferimento: docs/COMFY_ENGINE.md
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class WorkflowTemplate:
    """Template di un workflow ComfyUI, parametrizzabile prima del submit.

    Carica il grafo JSON e (se presente) il mapping dei ruoli semantici
    affiancato. I setter mutano una copia in memoria; il file su disco
    non viene mai modificato. :meth:`build` restituisce una copia
    indipendente pronta per ``ComfyClient.submit``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.graph: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        self.mapping: dict[str, Any] = self._load_mapping()

    def _load_mapping(self) -> dict[str, Any]:
        map_path = self.path.with_suffix(".map.json")
        if map_path.exists():
            return json.loads(map_path.read_text(encoding="utf-8"))
        logger.debug("Nessun mapping affiancato per %s", self.path.name)
        return {}

    # --- Accesso a basso livello (per nodo/campo) --------------------

    def set_param(self, node_id: str, field: str, value: Any) -> None:
        """Imposta ``inputs[field] = value`` sul nodo indicato."""
        node = self.graph.get(node_id)
        if node is None:
            raise KeyError(
                f"Nodo '{node_id}' assente nel workflow {self.path.name}"
            )
        node.setdefault("inputs", {})[field] = value

    def get_param(self, node_id: str, field: str) -> Any:
        return self.graph[node_id]["inputs"][field]

    # --- Accesso per ruolo semantico (via mapping) -------------------

    def set_role(self, role: str, value: Any) -> None:
        """Imposta il valore del nodo/campo associato a un ruolo del mapping."""
        spec = self.mapping.get(role)
        if not spec or "node" not in spec or "field" not in spec:
            raise KeyError(
                f"Ruolo '{role}' non mappato in {self.path.stem}.map.json"
            )
        self.set_param(spec["node"], spec["field"], value)

    def set_seed(self, seed: int) -> None:
        self.set_role("seed", int(seed))

    def set_prompt(self, positive: str, negative: Optional[str] = None) -> None:
        self.set_role("positive_prompt", positive)
        if negative is not None:
            self.set_role("negative_prompt", negative)

    def set_dimensions(self, width: int, height: int) -> None:
        self.set_role("width", int(width))
        self.set_role("height", int(height))

    def set_input_image(self, node_id: str, image_path: str) -> None:
        """Imposta l'immagine di input di un nodo LoadImage.

        ComfyUI riferisce le immagini per nome file dentro la sua input
        dir, quindi salviamo solo il basename.
        """
        self.set_param(node_id, "image", Path(image_path).name)

    def set_lora(self, lora_path: str, weight: float) -> None:
        """Imposta nome file e peso (model + clip) del nodo LoRA mappato."""
        spec = self.mapping.get("lora")
        if not spec or "node" not in spec:
            raise KeyError(
                f"Nessun nodo LoRA mappato in {self.path.stem}.map.json"
            )
        node = spec["node"]
        self.set_param(node, spec.get("name_field", "lora_name"), Path(lora_path).name)
        self.set_param(node, spec.get("model_weight_field", "strength_model"), float(weight))
        self.set_param(node, spec.get("clip_weight_field", "strength_clip"), float(weight))

    def set_loras(self, specs: list[tuple[str, float]]) -> None:
        """Impila più LoRA fra il checkpoint e i suoi consumatori.

        Il workflow ha un solo nodo LoRA (quello mappato): lo usiamo come
        modello per costruire una catena ``checkpoint → LoRA0 → LoRA1 → …``,
        rinstradando i consumatori (KSampler, CLIPSetLastLayer…) sull'ultimo
        anello. ``specs`` è una lista ordinata di ``(percorso, peso)``.

        Con lista vuota il nodo LoRA viene bypassato del tutto: i consumatori
        tornano a leggere model/clip direttamente dal checkpoint.
        """
        spec = self.mapping.get("lora")
        if not spec or "node" not in spec:
            raise KeyError(
                f"Nessun nodo LoRA mappato in {self.path.stem}.map.json"
            )
        head_id = spec["node"]
        head = self.graph[head_id]
        name_field = spec.get("name_field", "lora_name")
        model_w_field = spec.get("model_weight_field", "strength_model")
        clip_w_field = spec.get("clip_weight_field", "strength_clip")

        # Sorgenti a monte (model/clip) che alimentavano il nodo LoRA originale.
        up_model = head["inputs"]["model"]
        up_clip = head["inputs"]["clip"]

        if not specs:
            # Bypass: i consumatori del nodo LoRA tornano al checkpoint.
            self._rewire_slot(head_id, 0, up_model, exclude={head_id})
            self._rewire_slot(head_id, 1, up_clip, exclude={head_id})
            del self.graph[head_id]
            return

        # Il primo LoRA riusa il nodo esistente; i successivi sono cloni.
        path0, w0 = specs[0]
        head["inputs"][name_field] = Path(path0).name
        head["inputs"][model_w_field] = float(w0)
        head["inputs"][clip_w_field] = float(w0)
        head["inputs"]["model"] = up_model
        head["inputs"]["clip"] = up_clip

        chain = [head_id]
        prev = head_id
        for i, (path, weight) in enumerate(specs[1:], start=1):
            new_id = f"{head_id}_lora{i}"
            self.graph[new_id] = {
                "class_type": head["class_type"],
                "inputs": {
                    name_field: Path(path).name,
                    model_w_field: float(weight),
                    clip_w_field: float(weight),
                    "model": [prev, 0],
                    "clip": [prev, 1],
                },
            }
            chain.append(new_id)
            prev = new_id

        tail = chain[-1]
        if tail != head_id:
            # I consumatori puntavano a head_id: spostali sulla coda della catena
            # (senza toccare i link interni alla catena stessa).
            self._rewire_node(head_id, tail, exclude=set(chain))

    def _rewire_node(self, old_id: str, new_id: str, exclude: set[str]) -> None:
        """Rimpiazza ogni connessione ``[old_id, k]`` con ``[new_id, k]``."""
        for nid, node in self.graph.items():
            if nid in exclude:
                continue
            for field, val in node.get("inputs", {}).items():
                if isinstance(val, list) and len(val) == 2 and val[0] == old_id:
                    node["inputs"][field] = [new_id, val[1]]

    def _rewire_slot(self, old_id: str, slot: int, new_ref, exclude: set[str]) -> None:
        """Rimpiazza ``[old_id, slot]`` con ``new_ref`` (un riferimento intero)."""
        for nid, node in self.graph.items():
            if nid in exclude:
                continue
            for field, val in node.get("inputs", {}).items():
                if (
                    isinstance(val, list)
                    and len(val) == 2
                    and val[0] == old_id
                    and val[1] == slot
                ):
                    node["inputs"][field] = list(new_ref)

    # --- Output ------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Restituisce una copia profonda del grafo pronta per il submit."""
        return copy.deepcopy(self.graph)
