"""Diagnosi euristica dei crash di ComfyUI dal log.

Quando ComfyUI muore (processo terminato o WebSocket chiuso a metà
esecuzione), l'utente vede solo "possibile crash, controlla il log". Qui
riconosciamo i pattern noti nella coda del log e restituiamo un messaggio
con causa + rimedio pronto da seguire, invece del traceback grezzo.

Riferimento: docs/COMFY_ENGINE.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Quanti byte leggere dalla fine del file: basta per il blocco di traceback
# dell'ultimo crash, senza caricare log da centinaia di MB in memoria.
_TAIL_BYTES = 8000

_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("os error 1455", "paging file is too small", "paging file too small"),
        (
            "ComfyUI è andato in crash per il pagefile di Windows: è troppo "
            "piccolo (o gestito automaticamente) per l'allocazione improvvisa "
            "richiesta dal caricamento del modello.\n\n"
            "Soluzione: imposta un pagefile fisso di almeno 48 GB, poi riavvia "
            "il PC. Da PowerShell come amministratore:\n\n"
            "  $cs = Get-WmiObject Win32_ComputerSystem\n"
            "  $cs.AutomaticManagedPagefile = $false; $cs.Put()\n"
            "  Get-WmiObject Win32_PageFileSetting | ForEach-Object { $_.Delete() }\n"
            "  New-WmiObject Win32_PageFileSetting -Arguments @{\n"
            "      Name = 'C:\\pagefile.sys'; InitialSize = 49152; MaximumSize = 49152\n"
            "  }\n\n"
            "Poi riavvia il PC — dopo il riavvio il pagefile è pre-allocato e "
            "fisso, niente più crash da allocazione improvvisa."
        ),
    ),
    (
        ("access violation",),
        (
            "ComfyUI è andato in crash per esaurimento della RAM di sistema "
            "durante il caricamento del modello.\n\n"
            "Il profilo memoria prudente per PC con poca RAM dovrebbe essere "
            "già attivo. Se il crash persiste: chiudi altre app pesanti "
            "(browser, editor) prima di generare, oppure sistema il pagefile "
            "(vedi il messaggio dedicato se compare)."
        ),
    ),
    (
        ("out of memory", "cuda out of memory", "cuda error"),
        (
            "ComfyUI è andato in crash per mancanza di VRAM (memoria della "
            "scheda video).\n\n"
            "Prova a ridurre la risoluzione, generare una sola immagine per "
            "volta, o impostare la modalità VRAM su 'lowvram' nelle "
            "impostazioni."
        ),
    ),
)


def diagnose(log_path: Path) -> Optional[str]:
    """Legge la coda del log di ComfyUI e riconosce cause di crash note.

    Ritorna un messaggio utente pronto (causa + rimedio) se un pattern è
    riconosciuto, altrimenti None — il chiamante ricade su un messaggio
    generico. Non solleva mai: log assente/illeggibile → None.
    """
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            tail = f.read().decode("utf-8", "replace").lower()
    except OSError:
        return None

    for markers, message in _PATTERNS:
        if any(m in tail for m in markers):
            return message
    return None
