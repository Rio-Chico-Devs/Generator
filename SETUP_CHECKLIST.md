# Setup Checklist — Vihente Forge

Guida per validare l'ambiente PRIMA di scrivere codice. Esegui in ordine.
Il rischio numero uno è scoprire tardi che la GPU non è vista da torch:
questa checklist lo previene.

---

## Come dare le info a Claude Code

Non serve raccogliere tutto a mano. In Claude Code, apri il terminale
nella cartella `vihente-forge` e scrivi semplicemente:

> Controlla la mia configurazione di sistema: GPU, VRAM, versione CUDA,
> versione Python e spazio disco. Dimmi se è adeguata per Vihente Forge
> (serve NVIDIA 8GB+ VRAM, CUDA 12.x, Python 3.11, ~50GB liberi).

Claude Code eseguirà lui stesso i comandi e ti dirà se sei a posto.

Se invece vuoi raccogliere le info da solo prima, esegui i comandi sotto.

---

## STEP 1 — Hardware GPU (CRITICO)

```powershell
nvidia-smi
```

Cosa devi vedere:
- Nome della tua GPU (es. "NVIDIA GeForce RTX 3060")
- Memoria totale (in alto, es. "12288 MiB" = 12 GB)
- "CUDA Version: 12.x" in alto a destra

Verdetto:
- VRAM >= 8 GB → OK
- CUDA Version 12.0+ → OK
- Comando non riconosciuto / nessuna GPU → STOP: driver NVIDIA mancanti.
  Installa i driver prima di tutto il resto.

---

## STEP 2 — Python

```powershell
python --version
```

Verdetto:
- "Python 3.11.x" → OK
- "Python 3.12.x" o superiore → installa anche 3.11 (torch ha problemi su
  3.12 su Windows). Puoi tenerle entrambe.
- "Python 3.10" o inferiore → aggiorna a 3.11

---

## STEP 3 — Spazio disco

Windows:
```powershell
Get-PSDrive C
```
Linux:
```bash
df -h ~
```

Verdetto: servono almeno 50 GB liberi (10 ambiente + ~22 modelli + lavoro).

---

## STEP 4 — Sistema operativo

Lo sai già, ma per completezza:
- Windows 10/11 → supportato (target primario)
- Linux Ubuntu 22.04+ → supportato
- macOS → solo CPU, no training, sconsigliato per questo progetto

---

## STEP 5 — Crea l'ambiente

Segui docs/DEVELOPMENT.md alla lettera. In sintesi (Windows):

```powershell
cd <cartella vihente-forge>
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## STEP 6 — Verifica che torch veda la GPU (CRITICO)

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Verdetto:
- "CUDA: True" + nome GPU → tutto pronto
- "CUDA: False" → torch non vede la GPU. Cause comuni: torch CPU-only
  installato per errore (reinstalla con --index-url cu121), o driver
  CUDA non corrispondenti.

---

## STEP 7 — Primo traguardo: app in mock

```powershell
python -m src --mock
```

Cosa deve succedere:
- Si apre una finestra PyQt scura con sidebar "Progetti"
- Puoi cliccare "+ Nuovo", dare un nome, e vedere il progetto creato
- NON serve GPU né modelli: il mock simula tutto

Se la finestra si apre → l'impianto base funziona. Sei pronto per la
Fase 0 (embedding ComfyUI reale).

Se non si apre → l'errore in console dice cosa manca (di solito una
dipendenza). Dallo a Claude Code che lo risolve.

---

## Riassunto requisiti

| Requisito | Minimo | Come verificare |
|---|---|---|
| GPU | NVIDIA 8GB+ VRAM | nvidia-smi |
| CUDA | 12.0+ | nvidia-smi |
| Python | 3.11.x | python --version |
| RAM | 16 GB | (Task Manager / free -h) |
| Disco | 50 GB liberi | Get-PSDrive C / df -h |
| OS | Win 10/11 o Linux | — |

---

## Prima frase da dare a Claude Code

Una volta verificato l'ambiente, apri Claude Code nella cartella e digli:

> Leggi HANDOFF.md e tutti i file in docs/. Poi verifica il mio ambiente
> con questa checklist (SETUP_CHECKLIST.md). L'obiettivo di oggi è far
> partire `python -m src --mock` con la finestra che si apre. Quando
> funziona, passiamo alla Fase 0.
