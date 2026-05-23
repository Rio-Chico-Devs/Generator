from __future__ import annotations

from src.utils.seed import resolve_seed


def test_resolve_seed_passthrough_for_nonneg():
    assert resolve_seed(0) == 0
    assert resolve_seed(12345) == 12345


def test_resolve_seed_random_for_negative():
    s = resolve_seed(-1)
    assert isinstance(s, int)
    assert 0 <= s < 2**32


def test_resolve_seed_random_varies():
    seeds = {resolve_seed(-1) for _ in range(20)}
    assert len(seeds) > 1
