"""Lettura dell'output del subprocess di training riga per riga.

sd-scripts (kohya) usa tqdm, che aggiorna la progress bar riscrivendo la
stessa riga con '\r' invece di emettere un newline. Iterando lo stdout
del subprocess con il line-splitting standard (solo '\n') gli aggiornamenti
di progresso si accumulerebbero in un'unica riga consegnata a fine epoca:
progress bar e grafico loss della UI resterebbero congelati per minuti.

iter_output_lines() splitta su ENTRAMBI '\r' e '\n', consegnando ogni
aggiornamento tqdm come riga separata in tempo reale.

Modulo puro (niente Qt): testabile senza PyQt6.
"""
from __future__ import annotations

from typing import IO, Iterator


def iter_output_lines(stream: IO[str]) -> Iterator[str]:
    """Itera le righe di uno stream testuale splittando su '\\r' e '\\n'.

    Le righe vuote (sequenze '\\r\\n' o righe di soli separatori) vengono
    saltate. L'eventuale resto nel buffer a EOF viene consegnato.
    """
    buf: list[str] = []
    while True:
        ch = stream.read(1)
        if ch == "":
            break
        if ch == "\r" or ch == "\n":
            if buf:
                yield "".join(buf)
                buf = []
        else:
            buf.append(ch)
    if buf:
        yield "".join(buf)
