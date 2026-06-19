"""Test per src/core/credits — sistema di crediti simulato.

Puri stdlib (no GPU/Qt). I test di persistenza isolano VFORGE_APP_DIR in
una tempdir così non scrivono nelle cartelle reali dell'utente.
"""
from __future__ import annotations

import os
import tempfile

from src.core.config import GenerationConfig, LoraSpec
from src.core.credits import (
    ConsumptionSummary,
    CostModel,
    CreditWallet,
    InsufficientCreditsError,
    estimate_cost,
)


# ---------------------------------------------------------------------------
# Modello di costo
# ---------------------------------------------------------------------------


def test_base_cost_follows_megapixels_times_steps():
    cfg = GenerationConfig(prompt="x", width=1000, height=1000, steps=10, batch_size=1)
    # 1.0 MP × 10 step × 0.08 = 0.80
    b = estimate_cost(cfg, CostModel())
    assert b.per_image == 0.80
    assert b.total == 0.80


def test_cost_scales_with_resolution_and_steps():
    small = estimate_cost(GenerationConfig(prompt="x", width=512, height=512, steps=20))
    big = estimate_cost(GenerationConfig(prompt="x", width=1024, height=1024, steps=20))
    # quadruplicando i pixel il costo base quadruplica
    assert big.per_image > small.per_image
    assert abs(big.per_image - small.per_image * 4) < 0.05


def test_lora_surcharge_added():
    cfg = GenerationConfig(
        prompt="x", width=1000, height=1000, steps=10,
        loras=[LoraSpec(path="a"), LoraSpec(path="b"), LoraSpec(path="c")],
    )
    b = estimate_cost(cfg, CostModel())
    # 0.80 base + 3×0.20 = 1.40
    assert b.per_image == 1.40
    assert any("LoRA ×3" in i.label for i in b.items)


def test_controlnet_and_ipadapter_surcharges():
    cfg = GenerationConfig(prompt="x", width=1000, height=1000, steps=10)
    b = estimate_cost(cfg, CostModel(), use_controlnet=True, use_ipadapter=True)
    # 0.80 + 1.00 + 1.00 = 2.80
    assert b.per_image == 2.80
    labels = [i.label for i in b.items]
    assert any("ControlNet" in l for l in labels)
    assert any("IP-Adapter" in l for l in labels)


def test_upscale_adds_cost_for_extra_pixels():
    cfg = GenerationConfig(prompt="x", width=1000, height=1000, steps=10)
    b = estimate_cost(cfg, CostModel(), upscale_factor=2.0)
    # base 0.80 + upscale: 1.0 MP × (4-1) × 1.00 = 3.00 → 3.80
    assert b.per_image == 3.80
    assert any("Upscale 2x" in i.label for i in b.items)


def test_minimum_charge_floor():
    cfg = GenerationConfig(prompt="x", width=64, height=64, steps=1)
    b = estimate_cost(cfg, CostModel())
    # costo grezzo trascurabile → minimo 0.10
    assert b.per_image == 0.10


def test_batch_multiplies_total():
    cfg = GenerationConfig(prompt="x", width=1000, height=1000, steps=10, batch_size=4)
    b = estimate_cost(cfg, CostModel())
    assert b.per_image == 0.80
    assert b.total == 3.20


def test_breakdown_explain_is_readable():
    cfg = GenerationConfig(
        prompt="x", width=1000, height=1000, steps=10,
        loras=[LoraSpec(path="a")], batch_size=2,
    )
    text = estimate_cost(cfg, CostModel(), upscale_factor=2.0).explain()
    assert "Base" in text
    assert "LoRA" in text
    assert "Upscale" in text
    assert "Totale" in text


# ---------------------------------------------------------------------------
# Wallet: grant / charge
# ---------------------------------------------------------------------------


def test_grant_increases_balance_and_logs():
    w = CreditWallet()
    entry = w.grant(100.0, reason="bonus benvenuto")
    assert w.balance == 100.0
    assert entry.kind == "grant"
    assert entry.balance_after == 100.0
    assert len(w.ledger) == 1


def test_charge_decreases_balance_and_logs():
    w = CreditWallet()
    w.grant(10.0)
    w.charge(3.5, reason="generazione")
    assert w.balance == 6.5
    assert w.ledger[-1].kind == "charge"
    assert w.ledger[-1].amount == 3.5


def test_charge_insufficient_raises_and_keeps_balance():
    w = CreditWallet()
    w.grant(2.0)
    try:
        w.charge(5.0)
        assert False, "doveva sollevare InsufficientCreditsError"
    except InsufficientCreditsError as e:
        assert e.needed == 5.0
        assert e.available == 2.0
    assert w.balance == 2.0  # invariato
    # solo il grant è nel ledger, nessun addebito fallito registrato
    assert sum(1 for x in w.ledger if x.kind == "charge") == 0


def test_can_afford():
    w = CreditWallet()
    w.grant(5.0)
    assert w.can_afford(5.0)
    assert w.can_afford(4.99)
    assert not w.can_afford(5.01)


def test_grant_and_charge_reject_non_positive():
    w = CreditWallet()
    for bad in (0, -1.0):
        try:
            w.grant(bad)
            assert False
        except ValueError:
            pass
        try:
            w.charge(bad)
            assert False
        except ValueError:
            pass


def test_charge_for_generation_uses_breakdown_total():
    w = CreditWallet()
    w.grant(50.0)
    cfg = GenerationConfig(prompt="x", width=1000, height=1000, steps=10, batch_size=2)
    b = estimate_cost(cfg, CostModel())  # 0.80 × 2 = 1.60
    entry = w.charge_for_generation(b, reason="ricetta base")
    assert entry.amount == 1.60
    assert w.balance == 48.40
    assert entry.metadata["batch_size"] == 2
    assert entry.metadata["items"]  # scomposizione salvata


# ---------------------------------------------------------------------------
# Report di consumo
# ---------------------------------------------------------------------------


def test_consumption_summary_aggregates():
    w = CreditWallet()
    w.grant(100.0, reason="ricarica iniziale")
    w.charge(2.0, reason="ricetta base")
    w.charge(3.0, reason="ricetta base")
    w.charge(5.0, reason="personaggio in posa")
    s = w.consumption()
    assert isinstance(s, ConsumptionSummary)
    assert s.total_granted == 100.0
    assert s.total_spent == 10.0
    assert s.balance == 90.0
    assert s.charge_count == 3
    assert s.spent_by_reason["ricetta base"] == 5.0
    assert s.spent_by_reason["personaggio in posa"] == 5.0


def test_consumption_explain_lists_categories():
    w = CreditWallet()
    w.grant(20.0)
    w.charge(4.0, reason="upscale")
    text = w.consumption().explain()
    assert "Saldo attuale" in text
    assert "upscale" in text


# ---------------------------------------------------------------------------
# Persistenza
# ---------------------------------------------------------------------------


def test_wallet_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        old = os.environ.get("VFORGE_APP_DIR")
        os.environ["VFORGE_APP_DIR"] = d
        try:
            w = CreditWallet()
            w.grant(100.0, reason="iniziale")
            w.charge(7.25, reason="generazione")
            w.save()

            loaded = CreditWallet.load()
            assert loaded.balance == 92.75
            assert len(loaded.ledger) == 2
            assert loaded.ledger[0].reason == "iniziale"
            assert loaded.ledger[1].amount == 7.25
        finally:
            if old is None:
                del os.environ["VFORGE_APP_DIR"]
            else:
                os.environ["VFORGE_APP_DIR"] = old


def test_load_missing_file_returns_empty_wallet():
    with tempfile.TemporaryDirectory() as d:
        old = os.environ.get("VFORGE_APP_DIR")
        os.environ["VFORGE_APP_DIR"] = d
        try:
            w = CreditWallet.load()
            assert w.balance == 0.0
            assert w.ledger == []
        finally:
            if old is None:
                del os.environ["VFORGE_APP_DIR"]
            else:
                os.environ["VFORGE_APP_DIR"] = old


# ---------------------------------------------------------------------------
# Scenario completo (didattico)
# ---------------------------------------------------------------------------


def test_full_scenario_grant_generate_track():
    w = CreditWallet()
    w.grant(10.0, reason="bonus benvenuto")

    cfg = GenerationConfig(
        prompt="vf_iris_v1, masterpiece",
        width=512, height=768, steps=20,
        loras=[LoraSpec(path="style"), LoraSpec(path="char")],
        batch_size=1,
    )
    b = estimate_cost(cfg, CostModel(), upscale_factor=2.0, use_ipadapter=True)
    w.charge_for_generation(b, reason="personaggio in posa")

    s = w.consumption()
    assert s.total_granted == 10.0
    assert s.charge_count == 1
    assert s.balance == round(10.0 - b.total, 2)
    assert s.balance < 10.0  # ha consumato
