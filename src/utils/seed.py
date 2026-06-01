"""Gestione seed per riproducibilità."""
from __future__ import annotations

import secrets


def resolve_seed(seed: int) -> int:
    """Se seed < 0, genera uno random a 31 bit. Altrimenti restituisce così com'è.

    31 bit (0..2^31-1) invece di 32 garantisce che il seed rientri sempre nel
    range del QSpinBox (int32 con max 2_147_483_647). Un seed a 32 bit non
    rappresentabile in int32 verrebbe silenziosa­mente troncato dalla UI,
    impedendo la riproducibilità delle immagini.
    """
    if seed < 0:
        return secrets.randbits(31)
    return seed
