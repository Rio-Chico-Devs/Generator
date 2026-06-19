"""Rilevamento stato 'esaurito' — soluzioni insufficienti.

Riferimento: docs/GUIDED_TRAINING.md (Pilastro C)

Se in uno stesso step l'utente rifiuta tutti i candidati per K volte
consecutive, il sistema dichiara lo step 'esaurito': non è un fallimento ma
una diagnosi ("non ho abbastanza per fare quello che chiedi").

La logica opera sui DiaryEntry del diario, che sono la fonte di verità su
quante volte uno step è stato rifiutato di seguito.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.core.guidance.diary import DiaryEntry

DEFAULT_THRESHOLD = 3


@dataclass(frozen=True)
class ExhaustionReport:
    """Esito del controllo di esaurimento per uno step."""

    step_id: str
    exhausted: bool
    consecutive_rejections: int
    threshold: int


def consecutive_rejections(
    entries: list[DiaryEntry], session_id: str, step_id: str
) -> int:
    """Conta i rifiuti consecutivi più recenti per (session_id, step_id).

    Scorre le entry del diario in ordine cronologico, filtra quelle della
    sessione e dello step dati, e conta quanti rifiuti formano la striscia
    finale. Un'approvazione azzera il conteggio."""
    streak = 0
    for e in entries:
        if e.session_id != session_id or e.step_id != step_id:
            continue
        if e.is_approval:
            streak = 0
        else:
            streak += 1
    return streak


def check_exhaustion(
    entries: list[DiaryEntry],
    session_id: str,
    step_id: str,
    threshold: int = DEFAULT_THRESHOLD,
) -> ExhaustionReport:
    """True se lo step ha accumulato >= threshold rifiuti consecutivi."""
    count = consecutive_rejections(entries, session_id, step_id)
    return ExhaustionReport(
        step_id=step_id,
        exhausted=count >= threshold,
        consecutive_rejections=count,
        threshold=threshold,
    )


def diagnose(report: ExhaustionReport) -> str:
    """Messaggio diagnostico leggibile da mostrare nell'ExhaustedDialog."""
    total = report.consecutive_rejections
    return (
        f"Ho generato candidati per «{report.step_id}» {total} volte di "
        f"seguito senza che nessuno ti convincesse.\n\n"
        "Questo può significare che:\n"
        "→ Il dataset manca di esempi di questa tecnica\n"
        "→ Il LoRA attuale non ha visto abbastanza riferimenti adatti\n"
        "→ Va aggiunto un riferimento alla Libreria Tecniche prima di riprovare"
    )
