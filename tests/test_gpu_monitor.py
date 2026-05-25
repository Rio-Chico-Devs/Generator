"""Test per gpu_monitor: stato termico, SafetyConfig e ThermalGovernor.

Tutti i test sono puri: la lettura GPU è iniettata, quindi non servono né
una GPU NVIDIA né nvidia-smi installato.
"""
from __future__ import annotations

from src.utils.gpu_monitor import (
    CooldownOutcome,
    GpuSnapshot,
    SafetyConfig,
    ThermalGovernor,
    ThermalState,
    evaluate_safety,
    read_gpu,
)


# ---------------------------------------------------------------------------
# GpuSnapshot / ThermalState
# ---------------------------------------------------------------------------


def test_read_gpu_never_raises_and_marks_unavailable_without_nvidia():
    """In assenza di nvidia-smi (CI/cloud), read_gpu ritorna available=False."""
    snap = read_gpu()
    assert isinstance(snap, GpuSnapshot)
    # Non possiamo assumere la presenza di una GPU; verifichiamo solo che
    # il contratto regga (nessuna eccezione, tipo corretto).
    assert isinstance(snap.available, bool)


def test_thermal_state_thresholds():
    assert GpuSnapshot(available=True, temperature_c=50).thermal_state == ThermalState.COOL
    assert GpuSnapshot(available=True, temperature_c=75).thermal_state == ThermalState.NORMAL
    assert GpuSnapshot(available=True, temperature_c=85).thermal_state == ThermalState.WARM
    assert GpuSnapshot(available=True, temperature_c=90).thermal_state == ThermalState.HOT


def test_vram_pct_handles_zero_total():
    assert GpuSnapshot(available=True, vram_total_mb=0).vram_pct == 0


# ---------------------------------------------------------------------------
# SafetyConfig
# ---------------------------------------------------------------------------


def test_safety_config_defaults_are_laptop_conservative():
    cfg = SafetyConfig()
    assert cfg.pause_temp_c <= 80
    assert cfg.resume_temp_c < cfg.pause_temp_c
    assert cfg.cooldown_between_images_sec > 0


def test_safety_config_enforces_hysteresis():
    """Se resume >= pause, __post_init__ lo abbassa sotto pause."""
    cfg = SafetyConfig(pause_temp_c=78, resume_temp_c=80)
    assert cfg.resume_temp_c < cfg.pause_temp_c


# ---------------------------------------------------------------------------
# evaluate_safety (avvisi non bloccanti)
# ---------------------------------------------------------------------------


def test_evaluate_safety_pauses_above_threshold():
    cfg = SafetyConfig(warn_temp_c=72, pause_temp_c=78)
    snap = GpuSnapshot(available=True, temperature_c=80)
    should_pause, msg = evaluate_safety(snap, cfg)
    assert should_pause is True
    assert "80" in msg


def test_evaluate_safety_warns_in_band():
    cfg = SafetyConfig(warn_temp_c=72, pause_temp_c=78)
    snap = GpuSnapshot(available=True, temperature_c=74)
    should_pause, msg = evaluate_safety(snap, cfg)
    assert should_pause is False
    assert msg  # messaggio di avviso presente


def test_evaluate_safety_silent_when_cool():
    cfg = SafetyConfig(warn_temp_c=72, pause_temp_c=78)
    snap = GpuSnapshot(available=True, temperature_c=60)
    assert evaluate_safety(snap, cfg) == (False, "")


def test_evaluate_safety_silent_when_unavailable():
    snap = GpuSnapshot(available=False)
    assert evaluate_safety(snap, SafetyConfig()) == (False, "")


# ---------------------------------------------------------------------------
# ThermalGovernor.wait_until_safe
# ---------------------------------------------------------------------------


def _fixed_reader(snapshots):
    """read_fn che restituisce in sequenza gli snapshot dati; ripete l'ultimo."""
    seq = list(snapshots)

    def reader() -> GpuSnapshot:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return reader


def _fast_cfg(**kw) -> SafetyConfig:
    base = dict(pause_temp_c=78, resume_temp_c=68, poll_interval_sec=0.001,
                max_wait_sec=5.0)
    base.update(kw)
    return SafetyConfig(**base)


def test_governor_proceeds_when_unavailable():
    gov = ThermalGovernor(_fast_cfg(), read_fn=lambda: GpuSnapshot(available=False))
    assert gov.wait_until_safe() == CooldownOutcome.PROCEED


def test_governor_proceeds_when_already_cool():
    gov = ThermalGovernor(
        _fast_cfg(), read_fn=lambda: GpuSnapshot(available=True, temperature_c=60)
    )
    assert gov.wait_until_safe() == CooldownOutcome.PROCEED


def test_governor_waits_then_proceeds_after_cooldown():
    """Parte sopra pausa, poi scende sotto resume: deve procedere."""
    reader = _fixed_reader([
        GpuSnapshot(available=True, temperature_c=82),  # sopra pause -> attende
        GpuSnapshot(available=True, temperature_c=75),  # ancora sopra resume
        GpuSnapshot(available=True, temperature_c=66),  # sotto resume -> procede
    ])
    waits: list[int] = []
    gov = ThermalGovernor(_fast_cfg(), read_fn=reader)
    outcome = gov.wait_until_safe(on_wait=lambda s: waits.append(s.temperature_c))
    assert outcome == CooldownOutcome.PROCEED
    assert waits, "on_wait avrebbe dovuto essere chiamata durante l'attesa"


def test_governor_aborts_when_requested():
    gov = ThermalGovernor(
        _fast_cfg(), read_fn=lambda: GpuSnapshot(available=True, temperature_c=85)
    )
    outcome = gov.wait_until_safe(should_abort=lambda: True)
    assert outcome == CooldownOutcome.ABORTED


def test_governor_times_out_if_never_cools():
    gov = ThermalGovernor(
        _fast_cfg(max_wait_sec=0.05),
        read_fn=lambda: GpuSnapshot(available=True, temperature_c=90),
    )
    outcome = gov.wait_until_safe()
    assert outcome == CooldownOutcome.TIMEOUT


def test_governor_disabled_always_proceeds():
    gov = ThermalGovernor(
        SafetyConfig(enabled=False),
        read_fn=lambda: GpuSnapshot(available=True, temperature_c=99),
    )
    assert gov.wait_until_safe() == CooldownOutcome.PROCEED
