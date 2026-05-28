"""Dashboard 'Cosa ho imparato' — statistiche del Decision Diary (Pilastro C).

Mostra all'utente una panoramica delle proprie scelte:
- Quanti step approvati vs rifiutati per fase
- Motivi di rifiuto più frequenti
- Tecniche più usate
- Step con più problemi

Accessibile dalla sezione Training come sotto-tab "Sessioni Guidate".

Riferimento: docs/GUIDED_TRAINING.md (Pilastro C)
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.guidance.diary import diary_stats, load_diary
from src.core.project import Project

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Widget barra statistica (label + barra colorata + valore)
# ---------------------------------------------------------------------------


class _StatBar(QWidget):
    def __init__(self, label: str, value: int, max_value: int, color: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setFixedWidth(200)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        bar = QProgressBar()
        bar.setRange(0, max(max_value, 1))
        bar.setValue(value)
        bar.setTextVisible(False)
        bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        bar.setFixedHeight(16)
        layout.addWidget(bar, stretch=1)

        count = QLabel(str(value))
        count.setFixedWidth(40)
        count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(count)


# ---------------------------------------------------------------------------
# Sezione singola del dashboard
# ---------------------------------------------------------------------------


class _Section(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(lbl)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        layout.addWidget(sep)
        self._body = QVBoxLayout()
        layout.addLayout(self._body)

    def add_widget(self, w: QWidget) -> None:
        self._body.addWidget(w)

    def add_label(self, text: str) -> None:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        self._body.addWidget(lbl)


# ---------------------------------------------------------------------------
# Vista principale
# ---------------------------------------------------------------------------


class DiaryStatsView(QWidget):
    """Pannello statistiche Decision Diary per un progetto."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Optional[Project] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        bar = QHBoxLayout()
        title = QLabel("Sessioni Guidate — Cosa ho imparato")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        bar.addWidget(title)
        bar.addStretch()
        self._refresh_btn = QPushButton("↺  Aggiorna")
        self._refresh_btn.clicked.connect(self._reload)
        bar.addWidget(self._refresh_btn)
        outer.addLayout(bar)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        outer.addWidget(sep)

        # Area scrollabile
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(16)
        scroll.setWidget(self._content)

        self._show_empty()

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self._reload()

    def _reload(self) -> None:
        self._clear_layout()

        if self._project is None:
            self._show_empty()
            return

        if not self._project.diary_path.exists():
            self._layout.addWidget(
                _center_label(
                    "Nessuna sessione guidata ancora.\n"
                    "Vai su «Training Guidato» per iniziare."
                )
            )
            self._layout.addStretch()
            return

        entries = load_diary(self._project.diary_path)
        if not entries:
            self._layout.addWidget(
                _center_label("Il diario è vuoto — nessuna scelta registrata.")
            )
            self._layout.addStretch()
            return

        stats = diary_stats(entries)
        self._build_overview(stats)
        self._build_step_analysis(entries)
        self._build_rejection_reasons(stats)
        self._build_technique_usage(stats)
        self._layout.addStretch()

    # --- Sezioni ---------------------------------------------------------

    def _build_overview(self, stats: dict) -> None:
        sec = _Section("📊  Panoramica")
        approval_pct = int(stats["approval_rate"] * 100)
        color = "#5cb85c" if approval_pct >= 60 else "#d9a441" if approval_pct >= 30 else "#d96a6a"
        sec.add_label(
            f"Sessioni totali: {stats['total_sessions']}  |  "
            f"Decisioni totali: {stats['total_entries']}  |  "
            f"Tasso approvazione: {approval_pct}%  |  "
            f"Rigenerazioni: {stats['regeneration_count']}"
        )
        bar = _StatBar("Approvazioni", int(stats["approval_rate"] * stats["total_entries"]),
                       stats["total_entries"], color)
        sec.add_widget(bar)
        self._layout.addWidget(sec)

    def _build_step_analysis(self, entries) -> None:
        from collections import Counter
        step_approvals: Counter = Counter()
        step_rejections: Counter = Counter()
        for e in entries:
            if e.is_approval:
                step_approvals[e.step_id] += 1
            else:
                step_rejections[e.step_id] += 1

        all_steps = set(step_approvals) | set(step_rejections)
        if not all_steps:
            return

        sec = _Section("🔍  Analisi per step")
        max_total = max(step_approvals[s] + step_rejections[s] for s in all_steps)
        for step_id in sorted(all_steps):
            ok = step_approvals[step_id]
            ko = step_rejections[step_id]
            total = ok + ko
            color = "#5cb85c" if ok / total >= 0.6 else "#d9a441" if ok / total >= 0.3 else "#d96a6a"
            bar = _StatBar(f"{step_id}  ({ok}✓ / {ko}✗)", ok, max_total, color)
            sec.add_widget(bar)
        self._layout.addWidget(sec)

    def _build_rejection_reasons(self, stats: dict) -> None:
        reasons = stats["rejection_reasons"]
        if not reasons:
            return
        sec = _Section("💬  Motivi di rifiuto frequenti")
        top = reasons.most_common(8)
        max_count = top[0][1] if top else 1
        for reason, count in top:
            bar = _StatBar(reason, count, max_count, "#d9a441")
            sec.add_widget(bar)
        self._layout.addWidget(sec)

    def _build_technique_usage(self, stats: dict) -> None:
        usage = stats["technique_usage"]
        if not usage:
            return
        sec = _Section("🎨  Tecniche più usate (IP-Adapter)")
        top = usage.most_common(8)
        max_count = top[0][1] if top else 1
        for ref_id, count in top:
            # Cerca label nella libreria se disponibile
            label = ref_id
            if self._project:
                lib = None
                try:
                    from src.core.guidance.technique_library import TechniqueLibrary
                    lib = TechniqueLibrary.load(self._project.technique_library_path)
                    r = lib.get(ref_id)
                    if r:
                        label = f"{r.slot}: {r.label or r.ref_id[:8]}"
                except Exception:
                    pass
            bar = _StatBar(label, count, max_count, "#5b9bd5")
            sec.add_widget(bar)
        self._layout.addWidget(sec)

    # --- Utility ---------------------------------------------------------

    def _show_empty(self) -> None:
        self._layout.addWidget(
            _center_label("Nessun progetto attivo.")
        )
        self._layout.addStretch()

    def _clear_layout(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


def _center_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet("color: #888;")
    lbl.setWordWrap(True)
    return lbl
