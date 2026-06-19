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

Per i file rischiosi `build_report()` produce un resoconto leggibile che
spiega COSA è stato trovato e PERCHÉ è pericoloso, così l'utente può
decidere consapevolmente se procedere.

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
class Finding:
    """Un singolo import pericoloso rilevato in un pickle, con spiegazione."""

    module: str
    name: str
    opcode: str      # "GLOBAL" o "STACK_GLOBAL"
    category: str    # chiave categoria di rischio
    label: str       # etichetta leggibile della categoria
    why: str         # spiegazione del perché è pericoloso

    def target(self) -> str:
        return self.module if not self.name else f"{self.module}.{self.name}"

    def short(self) -> str:
        return f"{self.target()} [{self.label}]"


@dataclass
class ScanResult:
    path: Path
    verdict: ScanVerdict
    sha256: str
    file_format: str        # "safetensors" | "pickle_zip" | "pickle_raw" | "unknown"
    issues: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # __metadata__ del safetensors
    findings: list[Finding] = field(default_factory=list)  # import pericolosi nei pickle

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

    def report(self) -> str:
        """Resoconto leggibile da mostrare all'utente (vedi build_report)."""
        return build_report(self)


# ---------------------------------------------------------------------------
# Categorie di rischio — singola fonte di verità.
# La blacklist (moduli/nomi) è derivata da qui; ogni categoria porta con sé
# l'etichetta e la spiegazione mostrate nel report all'utente.
# ---------------------------------------------------------------------------

_RISK_CATEGORIES: dict[str, dict] = {
    "esecuzione_comandi": {
        "modules": {"os", "posix", "nt", "subprocess", "pty"},
        "names": {"system", "popen", "Popen"},
        "label": "Esecuzione di comandi di sistema",
        "why": (
            "Può eseguire qualsiasi comando sul computer (cancellare file, "
            "installare malware, esfiltrare dati). Un file di pesi non ne ha bisogno."
        ),
    },
    "rete": {
        "modules": {
            "socket", "socketserver", "http", "urllib", "ftplib", "smtplib",
            "requests", "httpx", "aiohttp", "paramiko", "fabric",
        },
        "label": "Accesso alla rete",
        "why": (
            "Può connettersi a server remoti per scaricare payload aggiuntivi "
            "o inviare i tuoi dati all'esterno senza che tu te ne accorga."
        ),
    },
    "esecuzione_codice": {
        "modules": {"builtins", "__builtin__", "__builtins__", "code", "codeop", "compileall"},
        "names": {"exec", "eval", "compile", "execfile"},
        "label": "Esecuzione di codice Python arbitrario",
        "why": (
            "Può eseguire codice Python costruito al volo, spesso usato per "
            "nascondere le reali intenzioni dietro stringhe offuscate."
        ),
    },
    "import_dinamico": {
        "modules": {"importlib", "runpy"},
        "names": {"__import__"},
        "label": "Import dinamico di moduli",
        "why": (
            "Carica moduli a runtime: tecnica comune per nascondere quali "
            "librerie pericolose vengono effettivamente usate."
        ),
    },
    "basso_livello": {
        "modules": {"ctypes", "cffi"},
        "label": "Accesso a memoria e librerie native",
        "why": (
            "Interagisce direttamente con la memoria e le librerie di sistema, "
            "aggirando le protezioni di Python."
        ),
    },
    "packaging": {
        "modules": {"pip", "setuptools", "distutils"},
        "label": "Installazione di pacchetti",
        "why": "Può scaricare e installare software aggiuntivo senza il tuo consenso.",
    },
    "deserializzazione": {
        "modules": {"pickle", "marshal"},
        "label": "Deserializzazione annidata",
        "why": (
            "Carica un secondo blocco di dati serializzati: tecnica di "
            "offuscamento per nascondere un payload dentro il payload."
        ),
    },
    "interprete": {
        "modules": {"sys", "multiprocessing"},
        "label": "Accesso agli interni dell'interprete",
        "why": "Manipola lo stato dell'interprete Python o avvia processi paralleli.",
    },
}

_DANGEROUS_MODULES = frozenset().union(
    *(cat.get("modules", frozenset()) for cat in _RISK_CATEGORIES.values())
)
_DANGEROUS_NAMES = frozenset().union(
    *(cat.get("names", frozenset()) for cat in _RISK_CATEGORIES.values())
)


def _classify(module: str, name: str) -> tuple[str, str, str]:
    """Mappa (module, name) alla categoria di rischio → (chiave, label, why)."""
    for key, cat in _RISK_CATEGORIES.items():
        if module in cat.get("modules", frozenset()) or name in cat.get("names", frozenset()):
            return key, cat["label"], cat["why"]
    return (
        "sconosciuto",
        "Import non riconosciuto",
        "Riferimento a un global non previsto in un file di pesi; origine incerta.",
    )


def _maybe_finding(module: str, name: str, opcode: str) -> Finding | None:
    """Crea un Finding se (module, name) è pericoloso, altrimenti None."""
    if module in _DANGEROUS_MODULES or name in _DANGEROUS_NAMES:
        category, label, why = _classify(module, name)
        return Finding(
            module=module, name=name, opcode=opcode,
            category=category, label=label, why=why,
        )
    return None


# ---------------------------------------------------------------------------
# Safetensors — parsing header-only (niente tensori caricati)
# ---------------------------------------------------------------------------


# Cap di sicurezza sull'header: un file corrotto/malevolo potrebbe dichiarare
# un header_len enorme (fino a 2^64). Gli header reali stanno sotto i pochi MB.
_MAX_HEADER_BYTES = 100 * 1024 * 1024  # 100 MB


def _read_header_region(path: Path) -> bytes:
    """Legge solo length(8) + header dal file, senza caricare i tensori.

    Per i checkpoint multi-GB questo evita di portare in RAM l'intero file:
    serve solo l'header per il verdetto safetensors. Se l'header dichiarato
    supera il cap o il file è troncato, ritorna quanto disponibile e lascia
    che :func:`_parse_safetensors` emetta l'issue corrispondente.
    """
    with path.open("rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            return length_bytes
        header_len = struct.unpack("<Q", length_bytes)[0]
        if header_len == 0 or header_len > _MAX_HEADER_BYTES:
            return length_bytes  # il parser segnala "header_length non plausibile"
        return length_bytes + f.read(header_len)


def _parse_safetensors(data: bytes, total_size: int) -> tuple[bool, list[str], dict]:
    """Valida l'header safetensors (``data`` = length+header) contro la
    dimensione totale del file (``total_size``), senza toccare i tensori.

    Ritorna (struttura_ok, issues, metadata).
    """
    issues: list[str] = []

    if len(data) < 8:
        return False, ["file troppo piccolo per essere safetensors (< 8 byte)"], {}

    (header_len,) = struct.unpack_from("<Q", data, 0)

    if header_len == 0 or header_len > _MAX_HEADER_BYTES:
        return False, [f"header_length non plausibile: {header_len}"], {}

    if len(data) < 8 + header_len or total_size < 8 + header_len:
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
    data_region = total_size - 8 - header_len
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


def _scan_pickle_bytes(data: bytes) -> tuple[list[Finding], list[str]]:
    """Variante a buffer di :func:`_scan_pickle_ops` (per il .pkl dentro lo ZIP)."""
    return _scan_pickle_ops(io.BytesIO(data))


def _scan_pickle_ops(stream) -> tuple[list[Finding], list[str]]:
    """Scansiona opcode pickle da uno stream senza eseguire nulla.

    Accetta un file-like leggibile (così un .pt grezzo multi-GB viene letto
    incrementalmente fino allo STOP, senza caricarlo tutto in RAM).

    Usa una finestra scorrevole sulle ultime 2 string literal per catturare
    sia GLOBAL (proto ≤ 3: "module name") che STACK_GLOBAL (proto ≥ 4:
    string + string + STACK_GLOBAL).

    Ritorna (findings, issues) dove findings sono import pericolosi (rendono
    il file DANGEROUS) e issues sono note diagnostiche (es. parse error).
    """
    findings: list[Finding] = []
    issues: list[str] = []
    recent_strings: list[str] = []

    try:
        ops = list(pickletools.genops(stream))
    except Exception as exc:
        return findings, [f"pickle non analizzabile: {exc}"]

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
            found = _maybe_finding(module, callable_name, "GLOBAL")
            if found:
                findings.append(found)

        elif name == "STACK_GLOBAL":
            # Formato moderno (proto ≥ 4): module e name sono in cima allo stack
            if len(recent_strings) >= 2:
                module, callable_name = recent_strings[-2], recent_strings[-1]
            elif len(recent_strings) == 1:
                module, callable_name = recent_strings[-1], ""
            else:
                module, callable_name = "?", "?"
            found = _maybe_finding(module, callable_name, "STACK_GLOBAL")
            if found:
                findings.append(found)

    return findings, issues


def _scan_as_pickle(path: Path, sha256: str) -> ScanResult:
    """Scansiona il file come pickle — raw o zip-wrapped (PyTorch .pt moderni).

    Legge dallo storage senza caricare l'intero file in RAM: lo ZIP è aperto
    direttamente dal path (solo il .pkl interno, piccolo, finisce in memoria),
    il pickle grezzo è letto in streaming dal file handle.
    """
    file_format = "pickle_raw"
    findings: list[Finding] = []
    issues: list[str] = []

    with path.open("rb") as f:
        magic = f.read(4)
        f.seek(0)

        if magic == b"PK\x03\x04":
            file_format = "pickle_zip"
            try:
                with zipfile.ZipFile(path) as zf:
                    pkl_names = [n for n in zf.namelist() if n.endswith(".pkl")]
                    if not pkl_names:
                        return ScanResult(
                            path=path, verdict=ScanVerdict.UNKNOWN,
                            sha256=sha256, file_format="zip_no_pkl",
                            issues=["archivio ZIP senza .pkl interno riconoscibile"],
                        )
                    findings, issues = _scan_pickle_bytes(zf.read(pkl_names[0]))
            except zipfile.BadZipFile as exc:
                return ScanResult(
                    path=path, verdict=ScanVerdict.UNKNOWN,
                    sha256=sha256, file_format="unknown",
                    issues=[f"ZIP malformato: {exc}"],
                )
        else:
            findings, issues = _scan_pickle_ops(f)

    verdict = ScanVerdict.DANGEROUS if findings else ScanVerdict.SUSPICIOUS_PICKLE
    # issues mostra prima le note diagnostiche, poi i findings in forma breve
    return ScanResult(
        path=path, verdict=verdict,
        sha256=sha256, file_format=file_format,
        issues=issues + [f.short() for f in findings],
        findings=findings,
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
        total_size = path.stat().st_size
    except OSError as exc:
        return ScanResult(
            path=path, verdict=ScanVerdict.UNKNOWN,
            sha256=sha256, file_format="unknown",
            issues=[f"impossibile leggere: {exc}"],
        )

    ext = path.suffix.lower()

    # --- Prova safetensors (solo header, niente tensori in RAM) ---
    if ext in _SAFETENSORS_EXT or ext in _AMBIGUOUS_EXT:
        try:
            header_region = _read_header_region(path)
        except OSError as exc:
            return ScanResult(
                path=path, verdict=ScanVerdict.UNKNOWN,
                sha256=sha256, file_format="unknown",
                issues=[f"impossibile leggere: {exc}"],
            )
        ok, st_issues, metadata = _parse_safetensors(header_region, total_size)
        if ok or ext in _SAFETENSORS_EXT:
            # File dichiarato .safetensors: ritorna SAFE o SUSPICIOUS
            verdict = ScanVerdict.SAFE if ok else ScanVerdict.SUSPICIOUS
            return ScanResult(
                path=path, verdict=verdict,
                sha256=sha256, file_format="safetensors",
                issues=st_issues, metadata=metadata,
            )
        # .bin non parsato come safetensors → prova come pickle

    # --- Pickle scan (streaming dal file, niente read_bytes dell'intero file) ---
    if ext in _PICKLE_EXT or ext in _AMBIGUOUS_EXT:
        try:
            return _scan_as_pickle(path, sha256)
        except OSError as exc:
            return ScanResult(
                path=path, verdict=ScanVerdict.UNKNOWN,
                sha256=sha256, file_format="unknown",
                issues=[f"impossibile leggere: {exc}"],
            )

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


# ---------------------------------------------------------------------------
# Report leggibile per l'utente
# ---------------------------------------------------------------------------

_FORMAT_LABELS = {
    "safetensors": "safetensors (solo dati, nessun codice)",
    "pickle_raw": "pickle grezzo (può contenere codice)",
    "pickle_zip": "pickle in archivio ZIP / PyTorch (può contenere codice)",
    "zip_no_pkl": "archivio ZIP non riconosciuto",
    "unknown": "sconosciuto",
}


def build_report(result: ScanResult) -> str:
    """Costruisce un resoconto leggibile da mostrare all'utente.

    Spiega cosa è stato trovato e perché è (o non è) pericoloso, così
    l'utente può decidere se procedere. La decisione resta sua: le proprietà
    `is_blocked` e `requires_override` indicano alla UI come comportarsi.
    """
    bar = "═" * 64
    sha = result.sha256[:32] + "…" if len(result.sha256) > 32 else result.sha256
    lines = [
        bar,
        "  REPORT DI SICUREZZA — file modello",
        bar,
        f"File:     {result.path.name}",
        f"Formato:  {_FORMAT_LABELS.get(result.file_format, result.file_format)}",
        f"SHA256:   {sha}",
        "",
    ]

    if result.verdict == ScanVerdict.SAFE:
        lines += [
            "VERDETTO: SICURO",
            "",
            "File safetensors strutturalmente valido. Contiene solo dati numerici",
            "(pesi del modello), nessun codice eseguibile. Caricamento sicuro.",
        ]
        st = f"{result.metadata.get('_tensor_count', '?')} tensori"
        lk = result.metadata.get("_lora_keys")
        if lk and lk != "0":
            st += f", {lk} chiavi LoRA riconosciute"
        lines += ["", f"Struttura: {st}."]

    elif result.verdict == ScanVerdict.DANGEROUS:
        lines += [
            "VERDETTO: PERICOLOSO — NON CARICARE",
            "",
            "Un file di pesi legittimo contiene solo numeri. Questo file include",
            "invece istruzioni che eseguono codice nel momento in cui viene aperto.",
            "Elementi rilevati:",
            "",
        ]
        for f in result.findings:
            lines += [
                f"  [!] {f.target()}   (opcode {f.opcode})",
                f"      Categoria: {f.label}",
                f"      Rischio:   {f.why}",
                "",
            ]
        lines += [
            "RACCOMANDAZIONE: non caricare questo file in ComfyUI/PyTorch.",
            "Eliminalo o tienilo in quarantena. Se l'hai scaricato, diffida",
            "della fonte: i modelli affidabili sono distribuiti in .safetensors.",
        ]

    elif result.verdict == ScanVerdict.SUSPICIOUS_PICKLE:
        lines += [
            "VERDETTO: SOSPETTO — richiede la tua conferma",
            "",
            "Questo file usa il formato 'pickle', che PUÒ contenere codice",
            "eseguibile. La scansione NON ha trovato import pericolosi noti,",
            "ma l'analisi statica non è una garanzia assoluta: tecniche di",
            "offuscamento avanzate potrebbero eluderla.",
        ]
        if result.issues:
            lines += ["", "Note tecniche:"]
            lines += [f"  • {i}" for i in result.issues]
        lines += [
            "",
            "Se esiste, preferisci la versione .safetensors dello stesso modello.",
            "Procedi solo se ti fidi della fonte: serve la tua conferma esplicita.",
        ]

    elif result.verdict == ScanVerdict.SUSPICIOUS:
        lines += [
            "VERDETTO: SOSPETTO — anomalie strutturali",
            "",
            "Il file si presenta come safetensors ma la struttura è anomala:",
        ]
        lines += [f"  • {i}" for i in result.issues]
        lines += [
            "",
            "Potrebbe essere corrotto o manomesso. Verifica la provenienza",
            "prima di usarlo.",
        ]

    else:  # UNKNOWN
        lines += [
            "VERDETTO: FORMATO SCONOSCIUTO — bloccato",
            "",
            "Impossibile analizzare il file in sicurezza:",
        ]
        lines += [f"  • {i}" for i in result.issues]
        lines += ["", "In assenza di garanzie, non caricarlo."]

    lines.append(bar)
    return "\n".join(lines)
