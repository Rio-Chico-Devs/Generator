"""Gestione seed per riproducibilità."""
from __future__ import annotations

import secrets


def resolve_seed(seed: int) -> int:
    """Se seed < 0, genera uno random a 32 bit. Altrimenti restituisce così com'è."""
    if seed < 0:
        return secrets.randbits(32)
    return seed
