"""Test per src/utils/model_scan — analisi di sicurezza senza eseguire codice.

Tutti i test sono puri Python (stdlib). Crea pickle sintetici malevoli/puliti
e safetensors minimali in directory temporanee; non richiede GPU, Qt né Pillow.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import struct
import tempfile
import zipfile
from pathlib import Path

from src.utils.model_scan import ScanVerdict, build_report, scan_model_file


# ---------------------------------------------------------------------------
# Costruttori helper
# ---------------------------------------------------------------------------


def _save(tmpdir: str, name: str, data: bytes) -> Path:
    p = Path(tmpdir) / name
    p.write_bytes(data)
    return p


def _make_safetensors(
    tensors: dict[str, int] | None = None,
    metadata: dict | None = None,
    corrupt_offsets: bool = False,
) -> bytes:
    """Costruisce un file safetensors minimale in memoria.

    ``tensors`` mappa nome → numero di float32 (ogni float = 4 byte).
    """
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    data_bytes = b""
    offset = 0
    for name, n_floats in (tensors or {}).items():
        n_bytes = n_floats * 4
        end = 9_999_999_999 if corrupt_offsets else offset + n_bytes
        header[name] = {
            "dtype": "F32",
            "shape": [n_floats],
            "data_offsets": [offset, end],
        }
        data_bytes += b"\x00" * n_bytes
        offset += n_bytes
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + data_bytes


def _make_zip_pkl(pkl_bytes: bytes) -> bytes:
    """Wrappa un pickle in un archivio ZIP stile PyTorch .pt."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", pkl_bytes)
    return buf.getvalue()


class _Malicious:
    """Oggetto pickle che referenzia os.system — usato come LoRA malevolo finto."""

    def __reduce__(self):
        return (os.system, ("echo hacked",))


class _NetworkMalicious:
    """Oggetto pickle che referenzia un modulo di rete (socket)."""

    def __reduce__(self):
        import socket

        return (socket.socket, ())


# ---------------------------------------------------------------------------
# Safetensors — SAFE
# ---------------------------------------------------------------------------


def test_safe_empty_safetensors():
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "empty.safetensors", _make_safetensors()))
    assert r.verdict == ScanVerdict.SAFE
    assert r.is_safe_to_use
    assert not r.is_blocked
    assert r.file_format == "safetensors"


def test_safe_lora_safetensors_with_metadata():
    tensors = {
        "lora_unet_down.lora_up.weight": 8,
        "lora_unet_down.lora_down.weight": 8,
        "lora_unet_down.alpha": 1,
    }
    meta = {"ss_base_model": "SDXL", "trigger": "vf_iris"}
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(
            _save(d, "style.safetensors", _make_safetensors(tensors, meta))
        )
    assert r.verdict == ScanVerdict.SAFE
    assert r.metadata.get("trigger") == "vf_iris"
    assert r.metadata.get("_lora_keys") == "3"


def test_safetensors_sha256_is_64char_hex():
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(
            _save(d, "x.safetensors", _make_safetensors({"lora_up": 4}))
        )
    assert len(r.sha256) == 64
    assert all(c in "0123456789abcdef" for c in r.sha256)


# ---------------------------------------------------------------------------
# Safetensors — SUSPICIOUS (anomalie strutturali)
# ---------------------------------------------------------------------------


def test_suspicious_safetensors_corrupt_offsets():
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(
            _save(d, "bad.safetensors", _make_safetensors({"lora_up": 4}, corrupt_offsets=True))
        )
    assert r.verdict == ScanVerdict.SUSPICIOUS
    assert not r.is_safe_to_use
    assert any("data_offsets" in i or "fuori" in i for i in r.issues)


def test_suspicious_safetensors_truncated():
    # Header dichiara 1000 byte ma il file è più corto
    bad = struct.pack("<Q", 1000) + b"short"
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "trunc.safetensors", bad))
    assert r.verdict == ScanVerdict.SUSPICIOUS
    assert r.file_format == "safetensors"


# ---------------------------------------------------------------------------
# .bin che è un safetensors — deve essere rilevato come SAFE, non pickle
# ---------------------------------------------------------------------------


def test_bin_safetensors_detected_as_safetensors():
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(
            _save(d, "weights.bin", _make_safetensors({"lora_up": 4}))
        )
    assert r.verdict == ScanVerdict.SAFE
    assert r.file_format == "safetensors"


# ---------------------------------------------------------------------------
# Pickle — DANGEROUS (import pericolosi rilevati)
# ---------------------------------------------------------------------------


def test_dangerous_pickle_global_opcode_os_system():
    """Protocol 2 → opcode GLOBAL 'os\\nsystem'."""
    pkl = pickle.dumps(_Malicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "evil.pt", pkl))
    assert r.verdict == ScanVerdict.DANGEROUS
    assert r.is_blocked
    assert not r.requires_override
    # os.system.__module__ è "posix" su Linux/macOS, "nt" su Windows;
    # entrambi sono in _DANGEROUS_MODULES.
    assert any("os" in i or "posix" in i or "nt" in i for i in r.issues)


def test_dangerous_pickle_stack_global_os_system():
    """Protocol 4 → opcode STACK_GLOBAL con 'os' + 'system'."""
    pkl = pickle.dumps(_Malicious(), protocol=4)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "evil2.pt", pkl))
    assert r.verdict == ScanVerdict.DANGEROUS
    assert r.is_blocked


def test_dangerous_pickle_ckpt_extension():
    pkl = pickle.dumps(_Malicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "evil.ckpt", pkl))
    assert r.verdict == ScanVerdict.DANGEROUS


def test_dangerous_pickle_bin_extension():
    """Un .bin con pickle malevolo (non safetensors) → DANGEROUS."""
    pkl = pickle.dumps(_Malicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "evil.bin", pkl))
    assert r.verdict == ScanVerdict.DANGEROUS


def test_dangerous_zip_wrapped_pickle():
    """PyTorch .pt zip-wrapped con payload malevolo → DANGEROUS."""
    pkl = pickle.dumps(_Malicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "evil_zipped.pt", _make_zip_pkl(pkl)))
    assert r.verdict == ScanVerdict.DANGEROUS
    assert r.file_format == "pickle_zip"


# ---------------------------------------------------------------------------
# Pickle — SUSPICIOUS_PICKLE (pulito, richiede override)
# ---------------------------------------------------------------------------


def test_suspicious_pickle_clean_dict():
    """Pickle pulito (nessun import pericoloso) → richiede override."""
    pkl = pickle.dumps({"lora_up": [0.1, 0.2], "alpha": 1.0})
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "clean.pt", pkl))
    assert r.verdict == ScanVerdict.SUSPICIOUS_PICKLE
    assert not r.is_blocked
    assert r.requires_override


def test_suspicious_pickle_zip_wrapped_clean():
    """Pickle zip-wrapped pulito (stile torch.save) → SUSPICIOUS_PICKLE."""
    pkl = pickle.dumps({"state_dict": {"weight": [1.0, 2.0]}})
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "model.bin", _make_zip_pkl(pkl)))
    assert r.verdict == ScanVerdict.SUSPICIOUS_PICKLE
    assert r.file_format == "pickle_zip"


# ---------------------------------------------------------------------------
# UNKNOWN — formato non supportato
# ---------------------------------------------------------------------------


def test_unknown_unsupported_extension():
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "model.lora2", b"gibberish"))
    assert r.verdict == ScanVerdict.UNKNOWN
    assert r.is_blocked


# ---------------------------------------------------------------------------
# summary() e proprietà
# ---------------------------------------------------------------------------


def test_summary_contains_verdict_uppercase():
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "x.safetensors", _make_safetensors()))
    assert "SAFE" in r.summary()


def test_summary_dangerous_contains_issue():
    pkl = pickle.dumps(_Malicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "e.pt", pkl))
    s = r.summary()
    assert "DANGEROUS" in s
    assert "os" in s or "posix" in s or "nt" in s


# ---------------------------------------------------------------------------
# Findings strutturati e classificazione del rischio
# ---------------------------------------------------------------------------


def test_findings_populated_on_dangerous():
    pkl = pickle.dumps(_Malicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "e.pt", pkl))
    assert len(r.findings) >= 1
    f = r.findings[0]
    assert f.module in ("os", "posix", "nt")
    assert f.category == "esecuzione_comandi"
    assert f.opcode == "GLOBAL"


def test_findings_classify_network_module():
    pkl = pickle.dumps(_NetworkMalicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "net.pt", pkl))
    assert r.verdict == ScanVerdict.DANGEROUS
    assert any(f.category == "rete" for f in r.findings)


def test_clean_pickle_has_no_findings():
    pkl = pickle.dumps({"weight": [1.0, 2.0, 3.0]})
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "clean.pt", pkl))
    assert r.findings == []
    assert r.verdict == ScanVerdict.SUSPICIOUS_PICKLE


# ---------------------------------------------------------------------------
# build_report — resoconto leggibile per l'utente
# ---------------------------------------------------------------------------


def test_report_dangerous_explains_risk():
    pkl = pickle.dumps(_Malicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "evil.pt", pkl))
    report = build_report(r)
    assert "PERICOLOSO" in report
    assert "NON CARICARE" in report
    # deve includere la spiegazione del rischio, non solo il nome del modulo
    assert "Esecuzione di comandi di sistema" in report
    assert "Rischio:" in report
    # e l'hash per tracciabilità
    assert r.sha256[:16] in report


def test_report_suspicious_pickle_asks_confirmation():
    pkl = pickle.dumps({"weight": [1.0, 2.0]})
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "clean.pt", pkl))
    report = build_report(r)
    assert "SOSPETTO" in report
    assert "conferma" in report.lower()
    assert "safetensors" in report


def test_report_safe_reassures():
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "ok.safetensors", _make_safetensors({"lora_up": 4})))
    report = build_report(r)
    assert "SICURO" in report
    assert "nessun codice eseguibile" in report


def test_report_accessible_via_result_method():
    pkl = pickle.dumps(_Malicious(), protocol=2)
    with tempfile.TemporaryDirectory() as d:
        r = scan_model_file(_save(d, "evil.pt", pkl))
    assert r.report() == build_report(r)
