"""Sistema di training guidato (human-in-the-loop).

Riferimento: docs/GUIDED_TRAINING.md

Quattro pilastri:
- session:           pipeline step-by-step (Pilastro A)
- technique_library: riferimenti IP-Adapter per tecnica (Pilastro B)
- diary:             registro append-only delle scelte (Pilastro C)
- exhaustion:        rilevamento stato "soluzioni insufficienti"

Tutti i moduli sono privi di dipendenze Qt per essere testabili in isolamento.
"""
