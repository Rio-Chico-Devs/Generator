"""Test per src/utils/seed — generazione e validazione seed.

Il vincolo fondamentale: i seed devono stare in [0, 2^31-1] per essere
rappresentabili come int32 senza troncamento silenzioso nel QSpinBox.
"""
from __future__ import annotations

from src.utils.seed import resolve_seed

# Il QSpinBox di Qt usa int32: max = 2^31 - 1 = 2_147_483_647
_INT32_MAX = 2_147_483_647
_INT31_EXCLUSIVE = 2**31   # randbits(31) genera in [0, 2^31)


# ---------------------------------------------------------------------------
# Passthrough: seed non-negativi restituiti invariati
# ---------------------------------------------------------------------------


def test_passthrough_zero():
    assert resolve_seed(0) == 0


def test_passthrough_one():
    assert resolve_seed(1) == 1


def test_passthrough_typical():
    assert resolve_seed(12345) == 12345


def test_passthrough_large():
    assert resolve_seed(1_000_000_000) == 1_000_000_000


def test_passthrough_int32_max():
    """Il valore massimo di QSpinBox deve passare invariato."""
    assert resolve_seed(_INT32_MAX) == _INT32_MAX


def test_passthrough_does_not_randomize():
    """Lo stesso seed positivo deve sempre tornare identico."""
    for seed in [0, 42, 12345, _INT32_MAX]:
        assert resolve_seed(seed) == seed


# ---------------------------------------------------------------------------
# Generazione casuale: seed negativi → 31 bit
# ---------------------------------------------------------------------------


def test_random_for_minus_one():
    s = resolve_seed(-1)
    assert isinstance(s, int)
    assert 0 <= s < _INT31_EXCLUSIVE


def test_random_for_large_negative():
    s = resolve_seed(-999_999)
    assert isinstance(s, int)
    assert 0 <= s < _INT31_EXCLUSIVE


def test_random_for_minus_int32_max():
    s = resolve_seed(-_INT32_MAX)
    assert 0 <= s < _INT31_EXCLUSIVE


def test_random_in_qspinbox_range():
    """Tutti i seed generati devono rientrare nel range valido del QSpinBox."""
    for _ in range(1000):
        s = resolve_seed(-1)
        assert 0 <= s <= _INT32_MAX, f"seed {s} fuori range QSpinBox"


def test_random_never_exceeds_31_bits():
    """randbits(31) garantisce < 2^31; questo verifica l'intera catena."""
    for _ in range(500):
        s = resolve_seed(-1)
        assert s < _INT31_EXCLUSIVE, f"seed {s} >= 2^31 (fuori range int32 positivo)"


def test_random_is_non_negative():
    for _ in range(100):
        s = resolve_seed(-1)
        assert s >= 0, f"seed negativo generato: {s}"


def test_random_varies():
    """Il generatore non deve essere deterministico."""
    seeds = {resolve_seed(-1) for _ in range(50)}
    assert len(seeds) > 1


def test_all_negatives_give_valid_seed():
    for neg in [-1, -42, -100, -1_000_000, -_INT32_MAX]:
        s = resolve_seed(neg)
        assert 0 <= s <= _INT32_MAX, f"resolve_seed({neg}) = {s} fuori range"


def test_random_return_type_is_int():
    for _ in range(10):
        assert type(resolve_seed(-1)) is int
