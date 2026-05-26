"""Analisi di sicurezza dei file modello (LoRA, checkpoint, ausiliari).

Protegge prima che un file entri nel sistema ComfyUI/PyTorch:
  .safetensors  → parsing header-only (zero esecuzione di codice)
  .pt/.ckpt/.bin → scansione opcode pickle (senza eseguire) + flag su import pericolosi

Verdetti
--------
SAFE             — safetensors strutturalmente valido
SUSPICIOUS       — safetensors con anomalie strutturali (offset fuori bounds, JSON corrotto)
SUSPICIOUS_PICKLE — pickle senza import pericolosi (richiede override esplicito dell'utente)
DANGEROUS        — pickle con import pericolosi → blocca sempre, non caricare mai
UNKNOWN          — formato non riconosciuto o file non leggibile

Nessuna dipendenza esterna: stdlib pura (struct, json, pickletools, zipfile, hashlib).
"""
from __future__ import annotations

import hashlib
import io
import json
import pickletools
import struct
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Verdetti e risultato
# ---------------------------------------------------------------------------


class ScanVerdict(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    SUSPICIOUS_PICKLE = "suspicious_pickle"
    DANGEROUS = "dangerous"
    UNKNOWN = "unknown"


@dataclass
class ScanResult:
    path: Path
    verdict: ScanVerdict
    sha256: str
    file_format: str        # "safetensors" | "pickle_zip" | "pickle_raw" | "unknown"
    issues: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # __metadata__ del safetensors

    @property
    def is_safe_to_use(self) -> bool:
        return self.verdict == ScanVerdict.SAFE

    @property
    def requires_override(self) -> bool:
        """Pickle pulito: l'utente può procedere solo con conferma esplicita."""
        return self.verdict == ScanVerdict.SUSPICIOUS_PICKLE

    @property
    def is_blocked(self) -> bool:
        """DANGEROUS o UNKNOWN: non deve mai essere caricato."""
        return self.verdict in (ScanVerdict.DANGEROUS, ScanVerdict.UNKNOWN)

    def summary(self) -> str:
        label = self.verdict.value.upper()
        detail = "; ".join(self.issues) if self.issues else self.file_format
        return f"[{label}] {detail}"


# ---------------------------------------------------------------------------
# Blacklist import pericolosi
# ---------------------------------------------------------------------------

_DANGEROUS_MODULES = frozenset({
    "os", "posix", "nt",
    "subprocess",
    "sys",
    "socket", "socketserver", "http", "urllib", "ftplib", "smtplib",
    "builtins", "__builtin__", "__builtins__",
    "runpy", "importlib", "pty",
    "ctypes", "cffi",
    "code", "codeop", "compileall",
    "requests", "httpx", "aiohttp", "paramiko", "fabric",
    "multiprocessing",
    "distutils", "setuptools", "pip",
    "pickle", "marshal",   # ricaricare deserializzatori dentro un pickle
})

_DANGEROUS_NAMES = frozenset({
    "system", "popen",
    "exec", "eval", "compile", "execfile",
    "__import__",
    "Popen",
})


# ---------------------------------------------------------------------------
# Safetensors — parsing header-only (niente tensori caricati)
# ---------------------------------------------------------------------------


def _parse_safetensors(data: bytes) -> tuple[bool, list[str], dict]:
    """Legge solo l'header safetensors senza toccare i tensori.

    Ritorna (struttura_ok, issues, metadata).
    """
    issues: list[str] = []

    if len(data) < 8:
        return False, ["file troppo piccolo per essere safetensors (< 8 byte)"], {}

    (header_len,) = struct.unpack_from("<Q", data, 0)

    if header_len == 0 or header_len > 100 * 1024 * 1024:
        return False, [f"header_length non plausibile: {header_len}"], {}

    if len(data) < 8 + header_len:
        return False, ["file troncato: header_len supera i byte disponibili"], {}

    try:
        header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, [f"header JSON non valido: {exc}"], {}

    if not isinstance(header, dict):
        return False, ["header non è un oggetto JSON"], {}

    # Estrai __metadata__ utente (solo string → string, no eval)
    metadata: dict = {}
    raw_meta = header.get("__metadata__")
    if isinstance(raw_meta, dict):
        metadata = {str(k): str(v) for k, v in raw_meta.items()}

    # Valida offset tensori contro la regione dati del file
    data_region = len(data) - 8 - header_len
    tensor_keys = [k for k in header if k != "__metadata__"]

    for key in tensor_keys:
        spec = header[key]
        if not isinstance(spec, dict):
            issues.append(f"tensore '{key}': spec non è un oggetto")
            continue
        offsets = spec.get("data_offsets")
        if isinstance(offsets, list) and len(offsets) == 2:
            start, end = offsets
            if end > data_region:
                issues.append(
                    f"tensore '{key}': data_offsets [{start},{end}] "
                    f"fuori dalla regione dati ({data_region} byte)"
                )

    # Info diagnostiche (non sicurezza) nel metadata
    lora_count = sum(
        1 for k in tensor_keys
        if any(x in k.lower() for x in ("lora_up", "lora_down", "alpha", "lora_te", "lora_unet"))
    )
    metadata["_tensor_count"] = str(len(tensor_keys))
    metadata["_lora_keys"] = str(lora_count)

    return len(issues) == 0, issues, metadata


# ---------------------------------------------------------------------------
# Pickle — scansione opcode (niente esecuzione)
# ---------------------------------------------------------------------------


def _scan_pickle_bytes(data: bytes) -> tuple[list[str], bool]:
    """Scansiona opcode pickle senza eseguire nulla.

    Usa una finestra scorrevole sulle ultime 2 string literal per catturare
    sia GLOBAL (proto ≤ 3: "module\\nname") che STACK_GLOBAL (proto ≥ 4:
    string + string + STACK_GLOBAL).

    Ritorna (issues, has_dangerous).
    """
    issues: list[str] = []
    has_dangerous = False
    recent_strings: list[str] = []

    try:
        ops = list(pickletools.genops(io.BytesIO(data)))
    except Exception as exc:
        return [f"pickle non analizzabile: {exc}"], False

    for opcode, arg, _pos in ops:
        name = opcode.name

        # Aggiorna finestra string literal (necessaria per STACK_GLOBAL)
        if name in (
            "STRING", "SHORT_BINSTRING", "BINSTRING",
            "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
        ):
            recent_strings.append(str(arg))
            recent_strings = recent_strings[-2:]   # tieni solo gli ultimi 2

        elif name == "GLOBAL":
            # pickletools restituisce module e name separati da spazio o \n
            # (Python 3 su Linux usa 'posix system', non 'os\nsystem')
            parts = str(arg).split(None, 1)
            module = parts[0] if parts else ""
            callable_name = parts[1] if len(parts) > 1 else ""
            if module in _DANGEROUS_MODULES or callable_name in _DANGEROUS_NAMES:
                issues.append(f"import pericoloso: {module}.{callable_name}")
                has_dangerous = True

        elif name == "STACK_GLOBAL":
            # Formato moderno (proto ≥ 4): module e name sono in cima allo stack
            if len(recent_strings) >= 2:
                module, callable_name = recent_strings[-2], recent_strings[-1]
            elif len(recent_strings) == 1:
                module, callable_name = recent_strings[-1], ""
            else:
                module, callable_name = "?", "?"

            if module in _DANGEROUS_MODULES or callable_name in _DANGEROUS_NAMES:
                issues.append(
                    f"import pericoloso (STACK_GLOBAL): {module}.{callable_name}"
                )
                has_dangerous = True

    return issues, has_dangerous


def _scan_as_pickle(path: Path, data: bytes, sha256: str) -> ScanResult:
    """Scansiona data come pickle — raw o zip-wrapped (PyTorch .pt moderni)."""
    file_format = "pickle_raw"
    issues: list[str] = []
    has_dangerous = False

    if data[:4] == b"PK\x03\x04":
        file_format = "pickle_zip"
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                pkl_names = [n for n in zf.namelist() if n.endswith(".pkl")]
                if not pkl_names:
                    return ScanResult(
                        path=path, verdict=ScanVerdict.UNKNOWN,
                        sha256=sha256, file_format="zip_no_pkl",
                        issues=["archivio ZIP senza .pkl interno riconoscibile"],
                    )
                issues, has_dangerous = _scan_pickle_bytes(zf.read(pkl_names[0]))
        except zipfile.BadZipFile as exc:
            return ScanResult(
                path=path, verdict=ScanVerdict.UNKNOWN,
                sha256=sha256, file_format="unknown",
                issues=[f"ZIP malformato: {exc}"],
            )
    else:
        issues, has_dangerous = _scan_pickle_bytes(data)

    verdict = ScanVerdict.DANGEROUS if has_dangerous else ScanVerdict.SUSPICIOUS_PICKLE
    return ScanResult(
        path=path, verdict=verdict,
        sha256=sha256, file_format=file_format,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Entrypoint pubblico
# ---------------------------------------------------------------------------

_SAFETENSORS_EXT = frozenset({".safetensors"})
_PICKLE_EXT = frozenset({".ckpt", ".pt", ".pth"})
_AMBIGUOUS_EXT = frozenset({".bin"})   # può essere safetensors o pickle — rileva dal contenuto


def scan_model_file(path: str | Path) -> ScanResult:
    """Scansiona un file modello senza eseguirne il codice.

    Sequenza:
    1. Calcola SHA256.
    2. Per .safetensors / .bin: prova parsing header safetensors.
    3. Per .pt/.ckpt/.pth / .bin non-safetensors: scansione opcode pickle.
    4. Formato non riconosciuto → UNKNOWN (bloccato).
    """
    path = Path(path)
    sha256 = _sha256(path)

    try:
        data = path.read_bytes()
    except OSError as exc:
        return ScanResult(
            path=path, verdict=ScanVerdict.UNKNOWN,
            sha256=sha256, file_format="unknown",
            issues=[f"impossibile leggere: {exc}"],
        )

    ext = path.suffix.lower()

    # --- Prova safetensors ---
    if ext in _SAFETENSORS_EXT or ext in _AMBIGUOUS_EXT:
        ok, st_issues, metadata = _parse_safetensors(data)
        if ok or ext in _SAFETENSORS_EXT:
            # File dichiarato .safetensors: ritorna SAFE o SUSPICIOUS
            verdict = ScanVerdict.SAFE if ok else ScanVerdict.SUSPICIOUS
            return ScanResult(
                path=path, verdict=verdict,
                sha256=sha256, file_format="safetensors",
                issues=st_issues, metadata=metadata,
            )
        # .bin non parsato come safetensors → prova come pickle

    # --- Pickle scan ---
    if ext in _PICKLE_EXT or ext in _AMBIGUOUS_EXT:
        return _scan_as_pickle(path, data, sha256)

    # --- Formato sconosciuto → blocca ---
    return ScanResult(
        path=path, verdict=ScanVerdict.UNKNOWN,
        sha256=sha256, file_format="unknown",
        issues=[f"estensione '{ext}' non riconosciuta dal scanner"],
    )


def _sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""
