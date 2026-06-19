"""Interfaccia per l'estrazione scheletro DWPose da immagini.

In Fase 1 l'estrazione è delegata al nodo 'DWPose Estimator' di ComfyUI:
l'app chiama ComfyUI via API, riceve il JSON keypoints, e lo converte
in PoseData tramite extract_from_comfy_output().

In Fase 2 (TODO): implementazione standalone via onnxruntime + DWPose weights
per l'estrazione offline senza ComfyUI avviato.

Formato keypoints (COCO-17, normalizzato [0,1]):
    [[x, y, confidence], ...]  — 17 joint nell'ordine COCO standard
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PoseData:
    """Dati scheletro estratti da un'immagine."""

    keypoints: list = field(default_factory=list)
    bones: list = field(default_factory=list)
    confidence: float = 0.0
    image_width: int = 0
    image_height: int = 0

    def to_dict(self) -> dict:
        return {
            "keypoints": self.keypoints,
            "bones": self.bones,
            "confidence": self.confidence,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PoseData:
        return cls(
            keypoints=d.get("keypoints", []),
            bones=d.get("bones", []),
            confidence=float(d.get("confidence", 0.0)),
            image_width=int(d.get("image_width", 0)),
            image_height=int(d.get("image_height", 0)),
        )

    @property
    def is_valid(self) -> bool:
        return bool(self.keypoints) and self.confidence > 0.0

    def as_joints_dict(self) -> dict[int, list[float]]:
        """Converte i keypoints in formato dict {index: [x, y, conf]}
        per categorize_skeleton() in library.py."""
        return {i: kp for i, kp in enumerate(self.keypoints)}


class PoseExtractor:
    """Estrae scheletri DWPose da immagini via ComfyUI.

    Uso tipico (Fase 1):
        extractor = PoseExtractor()
        comfy_output = comfy_client.run_dwpose(image_path)
        pose_data = extractor.extract_from_comfy_output(comfy_output)
    """

    def extract_from_comfy_output(self, keypoints_json: dict) -> PoseData:
        """Converte l'output JSON del nodo DWPose di ComfyUI in PoseData.

        Formato atteso (OpenPose JSON da ComfyUI DWPose Estimator):
            {"people": [{"pose_keypoints_2d": [x, y, conf, x, y, conf, ...]}],
             "canvas_width": W, "canvas_height": H}

        Normalizza le coordinate in [0, 1] rispetto a canvas_width/canvas_height.
        """
        people = keypoints_json.get("people", [])
        if not people:
            return PoseData()

        w = max(int(keypoints_json.get("canvas_width", 1)), 1)
        h = max(int(keypoints_json.get("canvas_height", 1)), 1)

        raw = people[0].get("pose_keypoints_2d", [])
        keypoints = []
        for i in range(0, len(raw) - 2, 3):
            x = raw[i] / w
            y = raw[i + 1] / h
            conf = raw[i + 2]
            keypoints.append([x, y, conf])

        confidence = (
            sum(kp[2] for kp in keypoints) / len(keypoints) if keypoints else 0.0
        )
        return PoseData(
            keypoints=keypoints,
            confidence=confidence,
            image_width=w,
            image_height=h,
        )
