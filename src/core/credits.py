"""Sistema di crediti simulato — per capire come funziona un'economia a crediti.

Tre parti, volutamente trasparenti a scopo didattico:

1. MODELLO DI COSTO (`CostModel` + `estimate_cost`)
   Calcola quanto costa una generazione partendo dai suoi parametri.
   La logica è: il costo segue il lavoro di calcolo reale, cioè
   "quanti pixel × quanti step di denoising", più sovrapprezzi per le
   feature extra (LoRA, ControlNet, upscale). Ogni voce è esplicita:
   `CostBreakdown.explain()` mostra la scomposizione riga per riga.

2. WALLET (`CreditWallet`)
   Tiene il saldo e un registro movimenti (ledger). `grant()` eroga
   crediti (la "ricarica"), `charge()` li scala (la "spesa"). Ogni
   movimento è tracciato con timestamp, motivo e saldo risultante.

3. REPORT DI CONSUMO (`CreditWallet.consumption()`)
   Aggrega il ledger: totale erogato, totale speso, saldo, numero di
   generazioni, spesa per categoria.

Persistenza: ~/.vihente-forge/credits.json (stesso schema degli altri config).
Nessuna dipendenza esterna oltre alla stdlib.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.paths import get_app_data_dir

logger = logging.getLogger(__name__)


def _round(x: float) -> float:
    """Arrotonda a 2 decimali (i crediti si mostrano come 6.80, 1.70...)."""
    return round(x + 1e-9, 2)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# 1. MODELLO DI COSTO
# ===========================================================================


@dataclass
class CostModel:
    """Tariffe del calcolo costo. Modificabili per simulare economie diverse.

    Il cuore è `credits_per_megapixel_step`: il costo base è proporzionale a
    (megapixel dell'immagine × numero di step), cioè al lavoro di denoising.
    """

    credits_per_megapixel_step: float = 0.08   # tariffa base del denoising
    lora_surcharge: float = 0.20               # per ogni LoRA caricato
    controlnet_surcharge: float = 1.00         # se attiva la guida di posa
    ipadapter_surcharge: float = 1.00          # se attivo il riferimento volto
    upscale_credits_per_megapixel: float = 1.00  # per ogni MP aggiunto dall'upscale
    minimum_charge: float = 0.10               # costo minimo di qualsiasi generazione


@dataclass
class CostItem:
    """Una singola voce di costo, mostrata nella scomposizione."""

    label: str
    amount: float


@dataclass
class CostBreakdown:
    """Scomposizione trasparente del costo di una generazione."""

    items: list[CostItem]
    batch_size: int = 1
    minimum_charge: float = 0.10

    @property
    def raw_per_image(self) -> float:
        return _round(sum(i.amount for i in self.items))

    @property
    def per_image(self) -> float:
        """Costo per immagine, mai sotto il minimo."""
        return _round(max(self.raw_per_image, self.minimum_charge))

    @property
    def total(self) -> float:
        """Costo totale del batch."""
        return _round(self.per_image * self.batch_size)

    def explain(self) -> str:
        """Resoconto leggibile riga per riga (per UI o log didattico)."""
        width = 44
        lines = ["Stima costo generazione", "─" * width]
        for item in self.items:
            lines.append(f"  {item.label:<34}{item.amount:>8.2f}")
        if self.raw_per_image < self.minimum_charge:
            lines.append(f"  {'(costo minimo applicato)':<34}{'':>8}")
        lines.append("─" * width)
        lines.append(f"  {'Per immagine':<34}{self.per_image:>8.2f}")
        if self.batch_size > 1:
            lines.append(f"  {'× ' + str(self.batch_size) + ' immagini':<34}{'':>8}")
            lines.append(f"  {'Totale':<34}{self.total:>8.2f}")
        return "\n".join(lines)


def estimate_cost(
    config,
    cost_model: Optional[CostModel] = None,
    *,
    upscale_factor: float = 1.0,
    use_controlnet: bool = False,
    use_ipadapter: bool = False,
) -> CostBreakdown:
    """Stima il costo di una generazione dai suoi parametri.

    `config` è un GenerationConfig (o qualsiasi oggetto con width, height,
    steps, batch_size, loras). Le feature di pipeline non deducibili dal
    config (ControlNet/IPAdapter/upscale) si passano come flag espliciti.
    """
    model = cost_model or CostModel()
    items: list[CostItem] = []

    megapixels = (config.width * config.height) / 1_000_000
    base = model.credits_per_megapixel_step * megapixels * config.steps
    items.append(
        CostItem(f"Base ({megapixels:.2f} MP × {config.steps} step)", _round(base))
    )

    lora_count = len(getattr(config, "loras", []) or [])
    if lora_count:
        items.append(
            CostItem(f"LoRA ×{lora_count}", _round(lora_count * model.lora_surcharge))
        )

    if use_controlnet:
        items.append(CostItem("ControlNet (posa)", _round(model.controlnet_surcharge)))
    if use_ipadapter:
        items.append(CostItem("IP-Adapter (volto)", _round(model.ipadapter_surcharge)))

    if upscale_factor > 1.0:
        added_mp = megapixels * (upscale_factor**2 - 1)
        items.append(
            CostItem(
                f"Upscale {upscale_factor:g}x (+{added_mp:.2f} MP)",
                _round(added_mp * model.upscale_credits_per_megapixel),
            )
        )

    return CostBreakdown(
        items=items,
        batch_size=getattr(config, "batch_size", 1) or 1,
        minimum_charge=model.minimum_charge,
    )


# ===========================================================================
# 2. WALLET (saldo + registro movimenti)
# ===========================================================================


class InsufficientCreditsError(Exception):
    """Sollevata quando si tenta di spendere più del saldo disponibile."""

    def __init__(self, needed: float, available: float):
        self.needed = _round(needed)
        self.available = _round(available)
        super().__init__(
            f"Crediti insufficienti: servono {self.needed:.2f}, "
            f"disponibili {self.available:.2f}"
        )


@dataclass
class LedgerEntry:
    """Un movimento nel registro: erogazione o spesa."""

    timestamp: str
    kind: str            # "grant" | "charge"
    amount: float        # sempre positivo
    balance_after: float
    reason: str
    metadata: dict = field(default_factory=dict)


@dataclass
class CreditWallet:
    """Portafoglio crediti con registro movimenti e persistenza JSON."""

    balance: float = 0.0
    ledger: list[LedgerEntry] = field(default_factory=list)

    # --- operazioni ---

    def grant(self, amount: float, reason: str = "ricarica") -> LedgerEntry:
        """Eroga crediti (la 'ricarica'). Questa è la logica che ti interessa
        imparare: aumenta il saldo e registra il movimento."""
        if amount <= 0:
            raise ValueError("L'importo da erogare deve essere positivo")
        self.balance = _round(self.balance + amount)
        entry = LedgerEntry(
            timestamp=_now_iso(), kind="grant",
            amount=_round(amount), balance_after=self.balance, reason=reason,
        )
        self.ledger.append(entry)
        return entry

    def can_afford(self, amount: float) -> bool:
        return self.balance + 1e-9 >= amount

    def charge(
        self, amount: float, reason: str = "generazione",
        metadata: Optional[dict] = None,
    ) -> LedgerEntry:
        """Scala crediti (la 'spesa'). Solleva InsufficientCreditsError se il
        saldo non basta, lasciando il saldo invariato."""
        if amount <= 0:
            raise ValueError("L'importo da addebitare deve essere positivo")
        if not self.can_afford(amount):
            raise InsufficientCreditsError(needed=amount, available=self.balance)
        self.balance = _round(self.balance - amount)
        entry = LedgerEntry(
            timestamp=_now_iso(), kind="charge",
            amount=_round(amount), balance_after=self.balance, reason=reason,
            metadata=metadata or {},
        )
        self.ledger.append(entry)
        return entry

    def charge_for_generation(
        self, breakdown: CostBreakdown, reason: str = "generazione",
    ) -> LedgerEntry:
        """Comodità: addebita il totale di una CostBreakdown, salvando la
        scomposizione nei metadati del movimento."""
        return self.charge(
            breakdown.total, reason=reason,
            metadata={
                "per_image": breakdown.per_image,
                "batch_size": breakdown.batch_size,
                "items": [asdict(i) for i in breakdown.items],
            },
        )

    # --- report di consumo ---

    def consumption(self) -> "ConsumptionSummary":
        granted = _round(sum(e.amount for e in self.ledger if e.kind == "grant"))
        spent = _round(sum(e.amount for e in self.ledger if e.kind == "charge"))
        charge_count = sum(1 for e in self.ledger if e.kind == "charge")
        by_reason: dict[str, float] = {}
        for e in self.ledger:
            if e.kind == "charge":
                by_reason[e.reason] = _round(by_reason.get(e.reason, 0.0) + e.amount)
        return ConsumptionSummary(
            balance=self.balance,
            total_granted=granted,
            total_spent=spent,
            charge_count=charge_count,
            spent_by_reason=by_reason,
        )

    # --- persistenza ---

    def config_path(self) -> Path:
        return get_app_data_dir() / "credits.json"

    def save(self) -> None:
        p = self.config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "balance": self.balance,
            "ledger": [asdict(e) for e in self.ledger],
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "CreditWallet":
        p = get_app_data_dir() / "credits.json"
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ledger = [
                LedgerEntry(
                    timestamp=e.get("timestamp", ""),
                    kind=e.get("kind", "charge"),
                    amount=e.get("amount", 0.0),
                    balance_after=e.get("balance_after", 0.0),
                    reason=e.get("reason", ""),
                    metadata=e.get("metadata", {}),
                )
                for e in data.get("ledger", [])
            ]
            return cls(balance=data.get("balance", 0.0), ledger=ledger)
        except Exception as e:
            logger.warning("credits.json corrotto, parto da zero: %s", e)
            return cls()


@dataclass
class ConsumptionSummary:
    """Aggregato del consumo crediti."""

    balance: float
    total_granted: float
    total_spent: float
    charge_count: int
    spent_by_reason: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        width = 44
        lines = ["Riepilogo crediti", "─" * width]
        lines.append(f"  {'Erogati (totale)':<34}{self.total_granted:>8.2f}")
        lines.append(f"  {'Spesi (totale)':<34}{self.total_spent:>8.2f}")
        lines.append(f"  {'Saldo attuale':<34}{self.balance:>8.2f}")
        lines.append(f"  {'Generazioni pagate':<34}{self.charge_count:>8d}")
        if self.spent_by_reason:
            lines.append("─" * width)
            lines.append("  Spesa per categoria:")
            for reason, amt in sorted(
                self.spent_by_reason.items(), key=lambda kv: kv[1], reverse=True
            ):
                lines.append(f"    {reason:<32}{amt:>8.2f}")
        return "\n".join(lines)
