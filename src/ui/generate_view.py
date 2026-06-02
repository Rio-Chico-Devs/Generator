"""Schermata di generazione — l'equivalente locale del pannello tensor.art.

Layout:
    +---------------------------+-----------------------------------+
    |  Controlli (scroll)       |  Risultati                        |
    |  - Prompt + negativo      |  [progress bar / stato]           |
    |    (contatore token)      |                                   |
    |  - Modello base + LoRA    |  [griglia thumbnail]              |
    |  - Impostazioni           |                                   |
    |  - Upscaler               |                                   |
    |  - Crediti + Genera       |                                   |
    +---------------------------+-----------------------------------+

La logica di calcolo (token, costo, parametri) vive in
``src.core.generation_form`` ed è testata senza Qt. Qui c'è solo il wiring
dei widget e la gestione del ciclo di vita del RecipeWorker.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.credits import CreditWallet, InsufficientCreditsError
from src.core.generation_form import (
    SAMPLERS,
    SCHEDULERS,
    SIZE_PRESETS,
    UPSCALE_FACTORS,
    GenerationForm,
    token_status,
)
from src.core.project import Project
from src.core.recipes import RecipeId, get_recipe

logger = logging.getLogger(__name__)

_TOKEN_COLORS = {"ok": "#6a9a6a", "warn": "#d9a441", "over": "#d96a6a"}
_THUMB_W = 188
_GALLERY_COLS = 2


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _LoraSlot(QWidget):
    """Una riga LoRA: abilita, nome file, peso. Scansiona il file alla scelta."""

    changed = pyqtSignal()

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self._path: Optional[Path] = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.enabled = QCheckBox()
        self.enabled.setToolTip("Abilita questo LoRA")
        self.enabled.toggled.connect(self.changed)
        row.addWidget(self.enabled)

        self.name = QLabel(f"LoRA {index + 1}: (nessuno)")
        self.name.setMinimumWidth(120)
        row.addWidget(self.name, 1)

        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.setToolTip("Scegli un file LoRA")
        browse.clicked.connect(self._on_browse)
        row.addWidget(browse)

        self.weight = QSlider(Qt.Orientation.Horizontal)
        self.weight.setMinimum(0)
        self.weight.setMaximum(200)   # 0.00 – 2.00
        self.weight.setValue(80)      # 0.80
        self.weight.setFixedWidth(90)
        self.weight.valueChanged.connect(self._on_weight)
        row.addWidget(self.weight)

        self.weight_label = QLabel("0.80")
        self.weight_label.setFixedWidth(34)
        row.addWidget(self.weight_label)

    def _on_weight(self, v: int) -> None:
        self.weight_label.setText(f"{v / 100:.2f}")
        self.changed.emit()

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Scegli LoRA", "",
            "Modelli (*.safetensors *.pt *.ckpt *.bin *.pth)",
        )
        if not path:
            return
        if not self._scan_and_confirm(Path(path)):
            return
        self._path = Path(path)
        self.name.setText(f"LoRA: {self._path.name}")
        self.enabled.setChecked(True)
        self.changed.emit()

    def _scan_and_confirm(self, path: Path) -> bool:
        """Scansiona il LoRA con model_scan; blocca i pericolosi, chiede
        conferma per i pickle sospetti. Ritorna True se si può usare."""
        try:
            from src.utils.model_scan import scan_model_file
        except Exception:
            return True  # scanner non disponibile: non bloccare
        result = scan_model_file(path)
        if result.is_blocked:
            QMessageBox.critical(self, "File bloccato", result.report())
            return False
        if result.requires_override:
            choice = QMessageBox.warning(
                self, "File sospetto",
                result.report() + "\n\nCaricarlo comunque?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return choice == QMessageBox.StandardButton.Yes
        return True

    def is_active(self) -> bool:
        return self.enabled.isChecked() and self._path is not None

    def weight_value(self) -> float:
        return self.weight.value() / 100.0

    def spec(self) -> Optional[tuple[str, float]]:
        """(path, peso) se lo slot è attivo, altrimenti None."""
        if self.is_active() and self._path is not None:
            return (str(self._path), self.weight_value())
        return None


class _ResultGallery(QWidget):
    """Griglia scrollabile di thumbnail dei risultati."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._inner = QWidget()
        self._grid = QGridLayout(self._inner)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)

        self._count = 0

    def clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._count = 0

    def add_image(self, path: Path) -> None:
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedSize(_THUMB_W, _THUMB_W)
        label.setStyleSheet("border: 1px solid #2a2d36; border-radius: 4px;")
        pix = QPixmap(str(path))
        if not pix.isNull():
            label.setPixmap(
                pix.scaled(
                    _THUMB_W - 6, _THUMB_W - 6,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            label.setText(path.name)
        label.setToolTip(str(path))
        r, c = divmod(self._count, _GALLERY_COLS)
        self._grid.addWidget(label, r, c)
        self._count += 1


class GenerateView(QWidget):
    """Vista di generazione completa. Dipendenze iniettate dopo la creazione
    via set_project / set_comfy_client / set_wallet."""

    # Emesso quando il saldo cambia, così MainWindow può persistere il wallet.
    balance_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._client = None
        self._wallet: Optional[CreditWallet] = None
        self._worker = None
        self._safety = None
        self._gpu_read_fn = None
        self._pending_per_image = 0.0
        self._seed_locked = False
        self._last_used_seed: Optional[int] = None

        self._build_ui()
        self._update_cost()
        self._update_generate_enabled()

    # --- costruzione UI -------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # Colonna controlli (in uno scroll, può eccedere l'altezza)
        controls = QWidget()
        col = QVBoxLayout(controls)
        col.setSpacing(10)
        col.addWidget(self._build_prompt_group())
        col.addWidget(self._build_model_group())
        col.addWidget(self._build_settings_group())
        col.addWidget(self._build_upscaler_group())
        col.addWidget(self._build_credit_group())
        col.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setFixedWidth(380)
        root.addWidget(scroll)

        # Colonna risultati
        right = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right.addWidget(self.progress)

        self.status_label = QLabel("Pronto.")
        self.status_label.setStyleSheet("color: #8a8d96;")
        right.addWidget(self.status_label)

        self.gallery = _ResultGallery()
        right.addWidget(self.gallery, 1)
        root.addLayout(right, 1)

    def _build_prompt_group(self) -> QGroupBox:
        box = QGroupBox("Prompt")
        v = QVBoxLayout(box)

        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Descrivi cosa vuoi generare…")
        self.prompt.setFixedHeight(96)
        self.prompt.textChanged.connect(self._on_prompt_changed)
        v.addWidget(self.prompt)

        self.prompt_tokens = QLabel("0 / 75 token")
        self.prompt_tokens.setStyleSheet(f"color: {_TOKEN_COLORS['ok']};")
        v.addWidget(self.prompt_tokens)

        v.addWidget(QLabel("Negativo (da evitare)"))
        self.negative = QPlainTextEdit()
        self.negative.setPlaceholderText("low quality, worst quality, blurry…")
        self.negative.setFixedHeight(60)
        self.negative.textChanged.connect(self._on_prompt_changed)
        v.addWidget(self.negative)

        self.negative_tokens = QLabel("0 / 75 token")
        self.negative_tokens.setStyleSheet(f"color: {_TOKEN_COLORS['ok']};")
        v.addWidget(self.negative_tokens)
        return box

    def _build_model_group(self) -> QGroupBox:
        box = QGroupBox("Modello")
        v = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Base:"))
        self.model_combo = QComboBox()
        from src.core.catalog import CATALOG
        for entry in CATALOG.values():
            self.model_combo.addItem(entry.name, entry.id)
        row.addWidget(self.model_combo, 1)
        v.addLayout(row)

        self.lora_slots: list[_LoraSlot] = []
        for i in range(3):
            slot = _LoraSlot(i)
            slot.changed.connect(self._update_cost)
            self.lora_slots.append(slot)
            v.addWidget(slot)
        return box

    def _build_settings_group(self) -> QGroupBox:
        box = QGroupBox("Impostazioni")
        v = QVBoxLayout(box)

        # --- Preset dimensioni (sempre visibili) ---
        v.addWidget(QLabel("Proporzioni"))
        preset_row = QHBoxLayout()
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        for idx, preset in enumerate(SIZE_PRESETS):
            btn = QPushButton(preset.label)
            btn.setCheckable(True)
            btn.clicked.connect(
                lambda _checked, p=preset: self._apply_preset(p.width, p.height)
            )
            self.preset_group.addButton(btn, idx)
            preset_row.addWidget(btn)
        custom_btn = QPushButton("Custom")
        custom_btn.setCheckable(True)
        self.preset_group.addButton(custom_btn, len(SIZE_PRESETS))
        preset_row.addWidget(custom_btn)
        v.addLayout(preset_row)

        # Larghezza / altezza
        wh = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 2048)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(832)
        self.width_spin.valueChanged.connect(self._on_size_spin)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 2048)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(1216)
        self.height_spin.valueChanged.connect(self._on_size_spin)
        wh.addWidget(QLabel("L:"))
        wh.addWidget(self.width_spin)
        wh.addWidget(QLabel("A:"))
        wh.addWidget(self.height_spin)
        v.addLayout(wh)

        # Step (sempre visibile: il parametro più impattante)
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Step:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 150)
        self.steps_spin.setValue(30)
        self.steps_spin.valueChanged.connect(self._update_cost)
        steps_row.addWidget(self.steps_spin, 1)
        v.addLayout(steps_row)

        # Batch (sempre visibile)
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Immagini:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16)
        self.batch_spin.setValue(1)
        self.batch_spin.valueChanged.connect(self._update_cost)
        batch_row.addWidget(self.batch_spin, 1)
        v.addLayout(batch_row)

        # Seed + lock (sempre visibili)
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        # 2^31-1: massimo di QSpinBox (int32 firmato in C++). resolve_seed()
        # genera seeds in 0..2^31-1 proprio per restare in questo range.
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setToolTip("-1 = casuale ad ogni generazione")
        seed_row.addWidget(self.seed_spin, 1)
        self.seed_lock_btn = QToolButton()
        self.seed_lock_btn.setText("🔓")
        self.seed_lock_btn.setCheckable(True)
        self.seed_lock_btn.setToolTip(
            "Blocca il seed: riusa lo stesso seed ad ogni generazione.\n"
            "Se il seed è -1, viene catturato automaticamente dopo la prima generazione."
        )
        self.seed_lock_btn.toggled.connect(self._on_seed_lock_toggled)
        seed_row.addWidget(self.seed_lock_btn)
        v.addLayout(seed_row)

        seed_help = QLabel(
            "Il seed è il \"punto di partenza\" casuale dell'immagine. "
            "Bloccalo con 🔒 per ritrovare lo stesso risultato, poi usa "
            "«Varianti ×4» per esplorare piccole variazioni della stessa idea."
        )
        seed_help.setWordWrap(True)
        seed_help.setStyleSheet("color: #8a8d96; font-size: 10px;")
        v.addWidget(seed_help)

        # --- Toggle impostazioni avanzate ---
        self.adv_toggle = QToolButton()
        self.adv_toggle.setText("⚙  Avanzate ▸")
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.adv_toggle.setStyleSheet("QToolButton { border: none; color: #8a8d96; }")
        self.adv_toggle.toggled.connect(self._on_adv_toggled)
        v.addWidget(self.adv_toggle)

        # Widget avanzato (nascosto di default)
        self._adv_widget = QWidget()
        adv = QVBoxLayout(self._adv_widget)
        adv.setContentsMargins(0, 4, 0, 0)
        adv.setSpacing(6)

        ss = QHBoxLayout()
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(SAMPLERS)
        self.sampler_combo.setCurrentText("dpmpp_2m")
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(SCHEDULERS)
        self.scheduler_combo.setCurrentText("karras")
        ss.addWidget(QLabel("Sampler:"))
        ss.addWidget(self.sampler_combo, 1)
        ss.addWidget(self.scheduler_combo, 1)
        adv.addLayout(ss)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("CFG:"))
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 30.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setValue(7.0)
        cfg_row.addWidget(self.cfg_spin, 1)
        adv.addLayout(cfg_row)

        self._adv_widget.setVisible(False)
        v.addWidget(self._adv_widget)
        return box

    def _on_adv_toggled(self, visible: bool) -> None:
        self._adv_widget.setVisible(visible)
        self.adv_toggle.setText("⚙  Avanzate ▾" if visible else "⚙  Avanzate ▸")

    def _build_upscaler_group(self) -> QGroupBox:
        box = QGroupBox("Upscaler")
        v = QVBoxLayout(box)

        self.upscale_check = QCheckBox("Ingrandisci l'output")
        self.upscale_check.toggled.connect(self._update_cost)
        v.addWidget(self.upscale_check)

        factor_row = QHBoxLayout()
        factor_row.addWidget(QLabel("Fattore:"))
        self.upscale_factor_combo = QComboBox()
        for f in UPSCALE_FACTORS:
            if f == 1.0:
                continue
            self.upscale_factor_combo.addItem(f"{f:g}x", f)
        self.upscale_factor_combo.setCurrentText("2x")
        self.upscale_factor_combo.currentIndexChanged.connect(self._update_cost)
        factor_row.addWidget(self.upscale_factor_combo, 1)
        v.addLayout(factor_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modello:"))
        self.upscaler_model_combo = QComboBox()
        from src.core.catalog import AUX_CATALOG
        for m in AUX_CATALOG.values():
            if m.kind == "upscaler":
                self.upscaler_model_combo.addItem(m.name, m.id)
        if self.upscaler_model_combo.count() == 0:
            self.upscaler_model_combo.addItem("R-ESRGAN 4x+", "real-esrgan-x4plus")
        model_row.addWidget(self.upscaler_model_combo, 1)
        v.addLayout(model_row)
        return box

    def _build_credit_group(self) -> QGroupBox:
        box = QGroupBox("Crediti")
        v = QVBoxLayout(box)

        self.balance_label = QLabel("Saldo: —")
        v.addWidget(self.balance_label)

        self.cost_label = QLabel("Costo stimato: —")
        self.cost_label.setStyleSheet("color: #d9a441;")
        v.addWidget(self.cost_label)

        gen_row = QHBoxLayout()
        self.generate_btn = QPushButton("Genera")
        self.generate_btn.clicked.connect(self._on_generate)
        gen_row.addWidget(self.generate_btn, 2)

        self.variants_btn = QPushButton("Varianti ×4")
        self.variants_btn.setToolTip(
            "Genera 4 varianti con seed sequenziali (seed, seed+1, seed+2, seed+3).\n"
            "Richiede un seed fissato — usa il pulsante 🔒 accanto al seed."
        )
        self.variants_btn.clicked.connect(self._on_generate_variants)
        gen_row.addWidget(self.variants_btn, 1)
        v.addLayout(gen_row)

        self.cancel_btn = QPushButton("Annulla")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        v.addWidget(self.cancel_btn)
        return box

    # --- dipendenze iniettate ------------------------------------------

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        if project is not None:
            self.negative.setPlaceholderText(
                project.default_negative_prompt or "low quality, worst quality"
            )
            if project.base_model is not None:
                idx = self.model_combo.findData(project.base_model.id)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
            d = project.default_generation_params
            self.steps_spin.setValue(d.steps)
            self.cfg_spin.setValue(d.cfg_scale)
        self._update_generate_enabled()

    def set_comfy_client(self, client) -> None:
        self._client = client
        self._update_generate_enabled()

    def set_wallet(self, wallet: CreditWallet) -> None:
        self._wallet = wallet
        self._update_balance_label()
        self._update_generate_enabled()

    def set_safety(self, safety, gpu_read_fn=None) -> None:
        self._safety = safety
        self._gpu_read_fn = gpu_read_fn

    # --- raccolta stato / costo ----------------------------------------

    def _collect_form(self) -> GenerationForm:
        extra_loras = tuple(
            spec for s in self.lora_slots if (spec := s.spec()) is not None
        )
        # peso stile: primo LoRA attivo, altrimenti default progetto
        lora_weight = extra_loras[0][1] if extra_loras else 0.85
        factor = (
            self.upscale_factor_combo.currentData()
            if self.upscale_check.isChecked()
            else 1.0
        ) or 1.0
        return GenerationForm(
            prompt=self.prompt.toPlainText(),
            negative=self.negative.toPlainText(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            steps=self.steps_spin.value(),
            cfg=self.cfg_spin.value(),
            sampler=self.sampler_combo.currentText(),
            scheduler=self.scheduler_combo.currentText(),
            seed=self.seed_spin.value(),
            lora_weight=lora_weight,
            batch=self.batch_spin.value(),
            upscale_factor=float(factor),
            extra_lora_count=len(extra_loras),
            extra_loras=extra_loras,
        )

    def _on_prompt_changed(self) -> None:
        self._set_token_label(self.prompt_tokens, self.prompt.toPlainText())
        self._set_token_label(self.negative_tokens, self.negative.toPlainText())
        self._update_generate_enabled()

    def _set_token_label(self, label: QLabel, text: str) -> None:
        st = token_status(text)
        label.setText(st.label() + (f"  ({st.chunks} chunk)" if st.chunks > 1 else ""))
        label.setStyleSheet(f"color: {_TOKEN_COLORS[st.level]};")

    def _apply_preset(self, width: int, height: int) -> None:
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)
        self._update_cost()

    def _on_size_spin(self) -> None:
        # L'utente ha toccato i numeri: passa a "Custom"
        custom_id = len(SIZE_PRESETS)
        btn = self.preset_group.button(custom_id)
        if btn is not None:
            btn.setChecked(True)
        self._update_cost()

    def _update_cost(self) -> None:
        form = self._collect_form()
        breakdown = form.estimate()
        self.cost_label.setText(f"Costo stimato: {breakdown.total:.2f} crediti")
        self.cost_label.setToolTip(breakdown.explain())
        if breakdown.total > 0:
            self.generate_btn.setText(f"Genera  —  {breakdown.total:.2f}")
        self._update_generate_enabled()

    def _update_balance_label(self) -> None:
        if self._wallet is not None:
            self.balance_label.setText(f"Saldo: {self._wallet.balance:.2f} crediti")

    def refresh_balance(self) -> None:
        """Pubblico: chiamato da MainWindow dopo una ricarica esterna."""
        self._update_balance_label()

    def apply_metadata(self, meta: dict) -> None:
        """Pre-compila il form dai metadati di un'immagine (riusa da galleria).

        Accetta le chiavi del sidecar (positive, negative, width, height,
        steps, cfg, sampler, seed). Valori assenti o non validi sono ignorati."""
        if "positive" in meta:
            self.prompt.setPlainText(str(meta["positive"]))
        if "negative" in meta:
            self.negative.setPlainText(str(meta["negative"]))
        w, h = _as_int(meta.get("width")), _as_int(meta.get("height"))
        if w is not None and h is not None:
            self._apply_preset(w, h)
            self._sync_preset_buttons(w, h)
        if (steps := _as_int(meta.get("steps"))) is not None:
            self.steps_spin.setValue(steps)
        if (cfg := _as_float(meta.get("cfg"))) is not None:
            self.cfg_spin.setValue(cfg)
        if meta.get("sampler"):
            idx = self.sampler_combo.findText(str(meta["sampler"]))
            if idx >= 0:
                self.sampler_combo.setCurrentIndex(idx)
        if (seed := _as_int(meta.get("seed"))) is not None and seed >= 0:
            self.seed_spin.setValue(min(seed, self.seed_spin.maximum()))
        self._on_prompt_changed()
        self._update_cost()

    def _sync_preset_buttons(self, width: int, height: int) -> None:
        """Evidenzia il bottone preset che combacia, o 'Custom'."""
        from src.core.generation_form import preset_for

        preset = preset_for(width, height)
        target_id = len(SIZE_PRESETS)  # Custom
        if preset is not None:
            for idx, p in enumerate(SIZE_PRESETS):
                if p.key == preset.key:
                    target_id = idx
                    break
        btn = self.preset_group.button(target_id)
        if btn is not None:
            btn.setChecked(True)

    def _on_seed_lock_toggled(self, locked: bool) -> None:
        self._seed_locked = locked
        if locked:
            self.seed_lock_btn.setText("🔒")
            # Se il seed è ancora -1 e abbiamo catturato un seed reale, usalo subito.
            if self.seed_spin.value() == -1 and self._last_used_seed is not None:
                self.seed_spin.setValue(self._last_used_seed)
            self.seed_spin.setEnabled(False)
        else:
            self.seed_lock_btn.setText("🔓")
            self.seed_spin.setEnabled(True)
        self._update_generate_enabled()

    def _update_generate_enabled(self) -> None:
        running = self._worker is not None and self._worker.isRunning()
        ready = (
            self._project is not None
            and self._client is not None
            and self._wallet is not None
            and bool(self.prompt.toPlainText().strip())
            and not running
        )
        self.generate_btn.setEnabled(ready)
        # Varianti: attive solo con seed fissato (>= 0) e non in corso
        seed_fixed = self.seed_spin.value() >= 0
        self.variants_btn.setEnabled(ready and seed_fixed)

    # --- generazione ----------------------------------------------------

    def _on_generate_variants(self) -> None:
        """Genera 4 varianti con seed sequenziali (seed, seed+1, seed+2, seed+3)."""
        if self.seed_spin.value() < 0:
            QMessageBox.information(
                self, "Seed non fissato",
                "Imposta un seed specifico e attiva il lucchetto 🔒 prima di\n"
                "generare varianti.\n\n"
                "Suggerimento: genera un'immagine, poi clicca 🔒 per catturare\n"
                "il seed usato.",
            )
            return
        self._on_generate(batch_override=4)

    def _on_generate(self, batch_override: Optional[int] = None) -> None:
        if self._project is None or self._client is None or self._wallet is None:
            return
        form = self._collect_form()
        if batch_override is not None:
            form.batch = batch_override
        breakdown = form.estimate()
        if not self._wallet.can_afford(breakdown.total):
            QMessageBox.warning(
                self, "Crediti insufficienti",
                f"Servono {breakdown.total:.2f} crediti, "
                f"disponibili {self._wallet.balance:.2f}.\n\n"
                "Usa 'Ricarica crediti' per aggiungerne.",
            )
            return

        self._pending_per_image = breakdown.per_image
        self._last_used_seed = None  # reset: cattura il seed della nuova generazione
        self.gallery.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminato finché non arriva il primo step
        self.cancel_btn.setVisible(True)
        self.status_label.setText("Generazione in corso…")

        from src.workers.recipe_worker import RecipeWorker

        recipe = get_recipe(RecipeId.BASE)
        self._worker = RecipeWorker(
            recipe=recipe,
            user_params=form.to_user_params(),
            project=self._project,
            client=self._client,
            output_dir=self._project.gallery_dir,
            comfy_input_dir=None,
            safety=self._safety,
            gpu_read_fn=self._gpu_read_fn,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.image_ready.connect(self._on_image_ready)
        self._worker.cooling.connect(self._on_cooling)
        self._worker.error.connect(self._on_error)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.start()
        self._update_generate_enabled()

    def _on_progress(self, step: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(step)

    def _on_image_ready(self, path: Path) -> None:
        # Addebita il costo per ogni immagine effettivamente prodotta
        if self._wallet is not None and self._pending_per_image > 0:
            try:
                self._wallet.charge(self._pending_per_image, reason="ricetta base")
                self._update_balance_label()
                self.balance_changed.emit()
            except InsufficientCreditsError:
                logger.warning("Saldo esaurito durante la generazione")
        self.gallery.add_image(path)

        # Cattura il seed effettivo dal sidecar (serve per il lock).
        # Viene fatto solo per la prima immagine di un batch (quella che
        # stabilisce il seed base), così non sovrascriviamo con seed+1, seed+2 ecc.
        if self._last_used_seed is None:
            try:
                from src.core.gallery import load_sidecar
                sidecar = load_sidecar(path)
                seed_used = sidecar.get("seed")
                if isinstance(seed_used, int) and seed_used >= 0:
                    self._last_used_seed = seed_used
                    if self._seed_locked and self.seed_spin.value() == -1:
                        # Lock attivo ma seed era -1: riempi con il seed reale.
                        self.seed_spin.setEnabled(True)
                        self.seed_spin.setValue(seed_used)
                        self.seed_spin.setEnabled(False)
                    self._update_generate_enabled()
            except Exception:
                pass  # Non fatale: la generazione non si blocca

    def _on_cooling(self, temp: int, resume: int) -> None:
        self.status_label.setText(
            f"GPU calda ({temp}°C): attendo che scenda sotto {resume}°C…"
        )

    def _on_error(self, msg: str) -> None:
        self.status_label.setText(msg)
        self._finish_run()

    def _on_finished(self, paths: list) -> None:
        self.status_label.setText(f"Completato: {len(paths)} immagini.")
        self.progress.setVisible(False)
        self._finish_run()

    def _finish_run(self) -> None:
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        try:
            self._disconnect_worker()
        finally:
            self._worker = None
        self._update_generate_enabled()

    def _disconnect_worker(self) -> None:
        """Scollega tutti i segnali del worker corrente (se presente)."""
        if self._worker is None:
            return
        for sig, slot in (
            (self._worker.progress,    self._on_progress),
            (self._worker.image_ready, self._on_image_ready),
            (self._worker.cooling,     self._on_cooling),
            (self._worker.error,       self._on_error),
            (self._worker.finished_ok, self._on_finished),
        ):
            try:
                sig.disconnect(slot)
            except RuntimeError:
                pass  # già disconnesso (es. shutdown() chiamato prima)

    def shutdown(self) -> None:
        """Ferma il worker in corso; chiamato da MainWindow.closeEvent.

        Disconnette i segnali prima di wait() così eventuali signal in coda
        non raggiungono slot di una view in fase di distruzione.
        """
        if self._worker is None:
            return
        self._disconnect_worker()
        self._worker.abort()
        self._worker.wait(5000)
        if self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self._worker = None

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self.status_label.setText("Interruzione richiesta…")
            self._worker.abort()
