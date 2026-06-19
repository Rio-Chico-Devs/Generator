"""Dialog crediti: mostra saldo e consumi, permette di erogare ('ricaricare').

È qui che si vede la logica di erogazione: il pulsante Ricarica chiama
CreditWallet.grant(), che aumenta il saldo e registra il movimento.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QDoubleSpinBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from src.core.credits import CreditWallet


class CreditDialog(QDialog):
    """Mostra saldo/consumi ed eroga crediti tramite grant()."""

    wallet_changed = pyqtSignal()

    def __init__(self, wallet: CreditWallet, parent=None) -> None:
        super().__init__(parent)
        self._wallet = wallet
        self.setWindowTitle("Crediti")
        self.resize(420, 380)

        v = QVBoxLayout(self)

        self.balance_label = QLabel()
        self.balance_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        v.addWidget(self.balance_label)

        # Riga ricarica
        recharge = QHBoxLayout()
        recharge.addWidget(QLabel("Eroga:"))
        self.amount = QDoubleSpinBox()
        self.amount.setRange(1.0, 1_000_000.0)
        self.amount.setValue(1000.0)
        self.amount.setSingleStep(100.0)
        recharge.addWidget(self.amount, 1)
        grant_btn = QPushButton("Ricarica")
        grant_btn.clicked.connect(self._on_grant)
        recharge.addWidget(grant_btn)
        v.addLayout(recharge)

        v.addWidget(QLabel("Consumi"))
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setStyleSheet("font-family: monospace;")
        v.addWidget(self.report, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        v.addWidget(buttons)

        self._refresh()

    def _on_grant(self) -> None:
        self._wallet.grant(self.amount.value(), reason="ricarica manuale")
        self._wallet.save()
        self.wallet_changed.emit()
        self._refresh()

    def _refresh(self) -> None:
        self.balance_label.setText(f"Saldo: {self._wallet.balance:.2f} crediti")
        self.report.setPlainText(self._wallet.consumption().explain())
