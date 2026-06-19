"""Test per src/studio/model_scanner — scansione modelli Studio Mode.

Costruisce file .safetensors sintetici in memoria (no GPU, no Qt).
Testa: read_safetensors_header, detect_family, extract_lora_metadata,
scan_dir, compatible_loras, is_compatible, ensure_studio_dirs.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from src.studio.model_scanner import (
    ModelFamily,
    ScannedModel,
    compatible_loras,
    detect_family,
    ensure_studio_dirs,
    extract_lora_metadata,
    is_compatible,
    read_safetensors_header,
    scan_dir,
    studio_dirs,
    studio_root,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _safetensors_bytes(tensors: dict | None = None, metadata: dict | None = None) -> bytes:
    """Costruisce i byte di un file .safetensors minimale in memoria."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    for name, spec in (tensors or {}).items():
        header[name] = spec
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_st(path: Path, tensors: dict | None = None, metadata: dict | None = None) -> Path:
    path.write_bytes(_safetensors_bytes(tensors, metadata))
    return path


def _scanned(family: ModelFamily, kind: str = "checkpoint") -> ScannedModel:
    return ScannedModel(
        path="x.safetensors", name="x", family=family, size_mb=100.0, kind=kind
    )


# ---------------------------------------------------------------------------
# read_safetensors_header
# ---------------------------------------------------------------------------


def test_read_header_valid_empty(tmp_path):
    p = _write_st(tmp_path / "empty.safetensors")
    h = read_safetensors_header(p)
    assert isinstance(h, dict)


def test_read_header_metadata_preserved(tmp_path):
    meta = {"ss_base_model_version": "sdxl_base_v1.0", "trigger": "iris"}
    p = _write_st(tmp_path / "lora.safetensors", metadata=meta)
    h = read_safetensors_header(p)
    assert h.get("__metadata__", {}).get("ss_base_model_version") == "sdxl_base_v1.0"
    assert h.get("__metadata__", {}).get("trigger") == "iris"


def test_read_header_tensor_keys_present(tmp_path):
    tensors = {"lora_up.weight": {"dtype": "F32", "shape": [8]}}
    p = _write_st(tmp_path / "model.safetensors", tensors=tensors)
    h = read_safetensors_header(p)
    assert "lora_up.weight" in h


def test_read_header_file_too_short(tmp_path):
    p = tmp_path / "short.safetensors"
    p.write_bytes(b"\x00\x01\x02")  # < 8 byte
    assert read_safetensors_header(p) == {}


def test_read_header_zero_length(tmp_path):
    p = tmp_path / "zero.safetensors"
    p.write_bytes(struct.pack("<Q", 0) + b"\x00" * 4)
    assert read_safetensors_header(p) == {}


def test_read_header_excessive_length(tmp_path):
    p = tmp_path / "huge.safetensors"
    p.write_bytes(struct.pack("<Q", 200 * 1024 * 1024) + b"\x00" * 8)
    assert read_safetensors_header(p) == {}


def test_read_header_invalid_json(tmp_path):
    raw = b"{ not valid json"
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert read_safetensors_header(p) == {}


def test_read_header_no_load_tensors(tmp_path):
    """L'header deve essere letto anche se il file è "troncato" dopo l'header
    (solo dati numerici assenti): la funzione legge SOLO i byte dell'header."""
    raw = json.dumps({"__metadata__": {"ok": "yes"}}).encode()
    # File con soli header byte e nessun dato tensore: valido
    p = tmp_path / "header_only.safetensors"
    p.write_bytes(struct.pack("<Q", len(raw)) + raw)
    h = read_safetensors_header(p)
    assert h.get("__metadata__", {}).get("ok") == "yes"


# ---------------------------------------------------------------------------
# detect_family
# ---------------------------------------------------------------------------


def test_detect_family_empty():
    assert detect_family({}) == ModelFamily.UNKNOWN


def test_detect_family_kohya_sdxl_base():
    h = {"__metadata__": {"ss_base_model_version": "sdxl_base_v1.0"}}
    assert detect_family(h) == ModelFamily.SDXL


def test_detect_family_kohya_xl_keyword():
    h = {"__metadata__": {"ss_base_model_version": "xl_v2"}}
    assert detect_family(h) == ModelFamily.SDXL


def test_detect_family_kohya_sdxl_keyword():
    h = {"__metadata__": {"ss_base_model_version": "sdxl"}}
    assert detect_family(h) == ModelFamily.SDXL


def test_detect_family_kohya_sd15_v1():
    h = {"__metadata__": {"ss_base_model_version": "v1"}}
    assert detect_family(h) == ModelFamily.SD15


def test_detect_family_kohya_sd15_sd_v1():
    h = {"__metadata__": {"ss_base_model_version": "sd_v1-5"}}
    assert detect_family(h) == ModelFamily.SD15


def test_detect_family_kohya_sd15_15_in_version():
    h = {"__metadata__": {"ss_base_model_version": "runwayml/stable-diffusion-v1-5"}}
    assert detect_family(h) == ModelFamily.SD15


def test_detect_family_kohya_flux():
    h = {"__metadata__": {"ss_base_model_version": "flux1-dev"}}
    assert detect_family(h) == ModelFamily.FLUX


def test_detect_family_flux_double_blocks_key():
    h = {"double_blocks.0.img_attn.norm.key_norm.scale": {"dtype": "F32", "shape": [128]}}
    assert detect_family(h) == ModelFamily.FLUX


def test_detect_family_flux_single_blocks_key():
    h = {"single_blocks.0.linear1.weight": {"dtype": "F32", "shape": [3072, 128]}}
    assert detect_family(h) == ModelFamily.FLUX


def test_detect_family_sdxl_cross_attention_2048():
    h = {
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight": {
            "dtype": "F32",
            "shape": [2048, 2048],
        }
    }
    assert detect_family(h) == ModelFamily.SDXL


def test_detect_family_sd15_cross_attention_768():
    h = {
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight": {
            "dtype": "F32",
            "shape": [320, 768],
        }
    }
    assert detect_family(h) == ModelFamily.SD15


def test_detect_family_sdxl_conditioner_embedders_key():
    h = {
        "conditioner.embedders.1.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight": {
            "dtype": "F32",
            "shape": [1280, 1280],
        }
    }
    assert detect_family(h) == ModelFamily.SDXL


def test_detect_family_metadata_beats_keys():
    """Il metadata kohya esplicito deve prevalere sulle euristiche sui tasti."""
    h = {
        "__metadata__": {"ss_base_model_version": "sd_v1-5"},
        "conditioner.embedders.1.something": {"dtype": "F32", "shape": [1]},
    }
    assert detect_family(h) == ModelFamily.SD15


def test_detect_family_unknown_no_hints():
    h = {"some_unrelated_key": {"dtype": "F32", "shape": [64]}}
    assert detect_family(h) == ModelFamily.UNKNOWN


def test_detect_family_only_metadata_key_no_version():
    h = {"__metadata__": {"trigger": "iris"}}
    # ss_base_model_version assente → non può determinare → UNKNOWN o fallback
    result = detect_family(h)
    assert isinstance(result, ModelFamily)


# ---------------------------------------------------------------------------
# extract_lora_metadata
# ---------------------------------------------------------------------------


def test_extract_lora_empty_header():
    triggers, hint = extract_lora_metadata({})
    assert triggers == []
    assert hint == ""


def test_extract_lora_base_version_hint():
    h = {"__metadata__": {"ss_base_model_version": "sdxl_base_v1.0"}}
    _, hint = extract_lora_metadata(h)
    assert hint == "sdxl_base_v1.0"


def test_extract_lora_sd_model_name_fallback():
    h = {"__metadata__": {"ss_sd_model_name": "MyCheckpoint"}}
    _, hint = extract_lora_metadata(h)
    assert hint == "MyCheckpoint"


def test_extract_lora_tag_frequency_top5(tmp_path):
    tags = {
        "ds1": {
            "iris": 100, "full_body": 90, "smile": 80,
            "outdoors": 70, "dress": 60, "rare_tag": 5,
        }
    }
    h = {"__metadata__": {"ss_tag_frequency": json.dumps(tags)}}
    triggers, _ = extract_lora_metadata(h)
    assert "iris" in triggers
    assert "full_body" in triggers
    assert len(triggers) <= 5
    assert "rare_tag" not in triggers


def test_extract_lora_tag_frequency_sorted_by_count():
    tags = {"ds": {"b": 10, "a": 50, "c": 30}}
    h = {"__metadata__": {"ss_tag_frequency": json.dumps(tags)}}
    triggers, _ = extract_lora_metadata(h)
    # "a" (50) deve essere prima di "b" (10)
    assert triggers.index("a") < triggers.index("b")


def test_extract_lora_invalid_tag_json():
    h = {"__metadata__": {"ss_tag_frequency": "{ not json"}}
    triggers, _ = extract_lora_metadata(h)
    assert triggers == []


def test_extract_lora_max_5_triggers():
    many = {f"tag_{i}": 100 - i for i in range(20)}
    h = {"__metadata__": {"ss_tag_frequency": json.dumps({"ds": many})}}
    triggers, _ = extract_lora_metadata(h)
    assert len(triggers) <= 5


def test_extract_lora_multi_dataset():
    tags = {"ds1": {"alpha": 10, "beta": 8}, "ds2": {"gamma": 9, "delta": 7}}
    h = {"__metadata__": {"ss_tag_frequency": json.dumps(tags)}}
    triggers, _ = extract_lora_metadata(h)
    assert len(triggers) <= 5


def test_extract_lora_no_metadata_key():
    h = {"lora_up.weight": {"dtype": "F32", "shape": [8]}}
    triggers, hint = extract_lora_metadata(h)
    assert triggers == []
    assert hint == ""


# ---------------------------------------------------------------------------
# scan_dir
# ---------------------------------------------------------------------------


def test_scan_dir_empty(tmp_path):
    assert scan_dir(tmp_path, "checkpoint") == []


def test_scan_dir_missing_dir(tmp_path):
    assert scan_dir(tmp_path / "nonexistent", "checkpoint") == []


def test_scan_dir_one_checkpoint(tmp_path):
    _write_st(tmp_path / "mymodel.safetensors")
    results = scan_dir(tmp_path, "checkpoint")
    assert len(results) == 1
    r = results[0]
    assert r.name == "mymodel"
    assert r.kind == "checkpoint"
    assert r.path == str(tmp_path / "mymodel.safetensors")


def test_scan_dir_lora_extracts_family(tmp_path):
    meta = {"ss_base_model_version": "sdxl_base_v1.0"}
    _write_st(tmp_path / "style.safetensors", metadata=meta)
    results = scan_dir(tmp_path, "lora")
    assert results[0].family == ModelFamily.SDXL
    assert results[0].kind == "lora"


def test_scan_dir_lora_extracts_trigger_words(tmp_path):
    tags = {"ds": {"iris": 100, "full_body": 90, "smile": 80, "dress": 70, "hair": 60}}
    meta = {"ss_tag_frequency": json.dumps(tags), "ss_base_model_version": "sdxl"}
    _write_st(tmp_path / "lora.safetensors", metadata=meta)
    results = scan_dir(tmp_path, "lora")
    assert "iris" in results[0].trigger_words


def test_scan_dir_ignores_non_safetensors(tmp_path):
    (tmp_path / "model.ckpt").write_bytes(b"fake")
    (tmp_path / "model.pt").write_bytes(b"fake")
    (tmp_path / "readme.txt").write_text("ciao")
    assert scan_dir(tmp_path, "checkpoint") == []


def test_scan_dir_multiple_files_sorted(tmp_path):
    for name in ["zebra.safetensors", "alpha.safetensors", "model.safetensors"]:
        _write_st(tmp_path / name)
    results = scan_dir(tmp_path, "checkpoint")
    names = [r.name for r in results]
    assert names == sorted(names)


def test_scan_dir_size_mb_non_negative(tmp_path):
    _write_st(tmp_path / "model.safetensors", metadata={"k": "v"})
    results = scan_dir(tmp_path, "checkpoint")
    assert results[0].size_mb >= 0.0


def test_scan_dir_metadata_stored(tmp_path):
    meta = {"foo": "bar", "ss_base_model_version": "sdxl"}
    _write_st(tmp_path / "lora.safetensors", metadata=meta)
    results = scan_dir(tmp_path, "lora")
    assert results[0].metadata.get("foo") == "bar"


def test_scan_dir_vae_kind(tmp_path):
    _write_st(tmp_path / "vae.safetensors")
    results = scan_dir(tmp_path, "vae")
    assert results[0].kind == "vae"


def test_scan_dir_embedding_kind(tmp_path):
    _write_st(tmp_path / "embedding.safetensors")
    results = scan_dir(tmp_path, "embedding")
    assert results[0].kind == "embedding"


def test_scan_dir_checkpoint_no_trigger_words(tmp_path):
    """I checkpoint non estraggono trigger words (solo i lora lo fanno)."""
    tags = {"ds": {"iris": 100}}
    meta = {"ss_tag_frequency": json.dumps(tags)}
    _write_st(tmp_path / "ckpt.safetensors", metadata=meta)
    results = scan_dir(tmp_path, "checkpoint")
    assert results[0].trigger_words == []


def test_scan_dir_unknown_family_fallback(tmp_path):
    _write_st(tmp_path / "mystery.safetensors")
    results = scan_dir(tmp_path, "checkpoint")
    assert results[0].family == ModelFamily.UNKNOWN


# ---------------------------------------------------------------------------
# is_compatible
# ---------------------------------------------------------------------------


def test_is_compatible_same_sdxl():
    assert is_compatible(_scanned(ModelFamily.SDXL), _scanned(ModelFamily.SDXL, "lora"))


def test_is_compatible_same_sd15():
    assert is_compatible(_scanned(ModelFamily.SD15), _scanned(ModelFamily.SD15, "lora"))


def test_is_compatible_same_flux():
    assert is_compatible(_scanned(ModelFamily.FLUX), _scanned(ModelFamily.FLUX, "lora"))


def test_is_compatible_sdxl_sd15_mismatch():
    assert not is_compatible(_scanned(ModelFamily.SDXL), _scanned(ModelFamily.SD15, "lora"))


def test_is_compatible_sd15_sdxl_mismatch():
    assert not is_compatible(_scanned(ModelFamily.SD15), _scanned(ModelFamily.SDXL, "lora"))


def test_is_compatible_flux_sdxl_mismatch():
    assert not is_compatible(_scanned(ModelFamily.FLUX), _scanned(ModelFamily.SDXL, "lora"))


def test_is_compatible_unknown_lora_always_true():
    for family in [ModelFamily.SD15, ModelFamily.SDXL, ModelFamily.FLUX]:
        assert is_compatible(_scanned(family), _scanned(ModelFamily.UNKNOWN, "lora")), (
            f"UNKNOWN lora deve essere compatibile con base {family}"
        )


# ---------------------------------------------------------------------------
# compatible_loras
# ---------------------------------------------------------------------------


def test_compatible_loras_empty():
    assert compatible_loras(_scanned(ModelFamily.SDXL), []) == []


def test_compatible_loras_all_compatible():
    base = _scanned(ModelFamily.SDXL)
    loras = [_scanned(ModelFamily.SDXL, "lora") for _ in range(3)]
    assert len(compatible_loras(base, loras)) == 3


def test_compatible_loras_filters_wrong_family():
    base = _scanned(ModelFamily.SDXL)
    loras = [
        _scanned(ModelFamily.SDXL, "lora"),
        _scanned(ModelFamily.SD15, "lora"),
        _scanned(ModelFamily.FLUX, "lora"),
    ]
    result = compatible_loras(base, loras)
    assert len(result) == 1
    assert result[0].family == ModelFamily.SDXL


def test_compatible_loras_unknown_passes_through():
    base = _scanned(ModelFamily.SD15)
    loras = [
        _scanned(ModelFamily.SD15, "lora"),
        _scanned(ModelFamily.UNKNOWN, "lora"),
        _scanned(ModelFamily.SDXL, "lora"),
    ]
    result = compatible_loras(base, loras)
    assert len(result) == 2
    families = {l.family for l in result}
    assert ModelFamily.SD15 in families
    assert ModelFamily.UNKNOWN in families


def test_compatible_loras_all_unknown():
    base = _scanned(ModelFamily.SD15)
    loras = [_scanned(ModelFamily.UNKNOWN, "lora") for _ in range(4)]
    assert len(compatible_loras(base, loras)) == 4


def test_compatible_loras_none_compatible():
    base = _scanned(ModelFamily.FLUX)
    loras = [_scanned(ModelFamily.SD15, "lora"), _scanned(ModelFamily.SDXL, "lora")]
    assert compatible_loras(base, loras) == []


# ---------------------------------------------------------------------------
# ensure_studio_dirs / studio_dirs / studio_root
# ---------------------------------------------------------------------------


def test_ensure_studio_dirs_creates_all():
    ensure_studio_dirs()
    for name, path in studio_dirs().items():
        assert path.exists(), f"Cartella studio '{name}' non creata: {path}"
        assert path.is_dir()


def test_studio_dirs_has_all_keys():
    assert set(studio_dirs().keys()) == {"checkpoint", "lora", "vae", "embedding"}


def test_studio_root_name():
    assert studio_root().name == "studio"


def test_studio_root_under_user_data():
    from src.utils.paths import get_user_data_dir
    assert studio_root() == get_user_data_dir() / "studio"


def test_studio_dirs_are_under_root():
    root = studio_root()
    for path in studio_dirs().values():
        assert str(path).startswith(str(root))


def test_ensure_studio_dirs_idempotent():
    ensure_studio_dirs()
    ensure_studio_dirs()  # seconda chiamata non deve sollevare eccezioni
    for path in studio_dirs().values():
        assert path.is_dir()
