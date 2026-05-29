"""Scritture atomiche su file.

Pattern: scrivi su un file temporaneo nella stessa directory, fai fsync,
poi rinomina con os.replace() (atomico su POSIX e Windows). Se il processo
crasha a metà, il file di destinazione resta intatto: o c'è la vecchia
versione completa, o la nuova versione completa, mai una via di mezzo
corrotta.

Usato per session.json, library.json e l'append al diary JSONL.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """Scrive `data` su `path` in modo atomico (temp + fsync + rename).

    Crea le directory genitrici se mancano. Solleva OSError se la scrittura
    o il rename falliscono (il chiamante decide se è fatale o best-effort).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Il temp DEVE stare nella stessa directory: os.replace è atomico solo
    # all'interno dello stesso filesystem.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Pulisci il temp in qualsiasi caso di errore (anche KeyboardInterrupt).
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_append_line(path: Path, line: str, encoding: str = "utf-8") -> None:
    """Appende una riga a un file JSONL in modo durevole.

    L'append puro non è atomico come il rename, ma con flush+fsync e una
    sola write() della riga completa (terminatore incluso) il rischio di
    riga troncata è minimo. load_diary() ripara comunque l'ultima riga
    eventualmente corrotta.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = line if line.endswith("\n") else line + "\n"
    with path.open("a", encoding=encoding) as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
