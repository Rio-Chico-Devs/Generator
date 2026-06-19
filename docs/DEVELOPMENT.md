# Sviluppo

## Due modalità di sviluppo

| Modalità | Quando | Cosa serve |
|---|---|---|
| **Logica / UI** | Tutti i giorni, senza GPU | Python 3.11, PyQt6, Pillow, pytest |
| **Generazione reale** | Test end-to-end con output veri | + ComfyUI installato + GPU NVIDIA 8 GB |

Il motore di generazione è **ComfyUI headless** (processo figlio locale).
L'app non importa torch né diffusers: parla con ComfyUI via HTTP/WebSocket.
In `--mock` tutto lo sviluppo UI funziona senza GPU e senza ComfyUI installato.

---

## Setup sviluppo logica / UI (senza GPU)

```bash
# 1. Python 3.11
python --version   # Python 3.11.x

# 2. Clone repo
git clone <repo> ~/dev/vihente-forge
cd ~/dev/vihente-forge

# 3. Virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 4. Dipendenze leggere (UI + test)
pip install PyQt6 Pillow pytest pytest-qt ruff black

# 5. Verifica
pytest tests/ -x
python -m src --mock --skip-model-check   # apre la finestra
```

Non serve CUDA, non serve torch, non serve ComfyUI per sviluppare e
testare la logica applicativa.

---

## Setup runtime completo (generazione reale)

Necessario solo per testare ricette end-to-end con output veri.

### Requisiti hardware
- **GPU**: NVIDIA 8+ GB VRAM (RTX 2060+, RTX 30xx/40xx)
- **RAM**: 16 GB minimo, 32 GB consigliato
- **Disco**: 50 GB liberi (ComfyUI + modelli + progetti)
- **OS**: Windows 10/11 o Ubuntu 22.04+

### Installazione ComfyUI

```bash
# Nella cartella dati utente (default: ~/Documents/Vihente\ Forge/engine/)
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
# Pinna la versione nota-funzionante (aggiornare con cautela)
git checkout <commit-hash-pinnato>

pip install -r requirements.txt   # installa torch CUDA qui

# Custom nodes necessari per le ricette
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack          # Adetailer
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus           # IP-Adapter
git clone https://github.com/Fannovel16/comfyui_controlnet_aux      # pose extraction
git clone https://github.com/huchenlei/ComfyUI-IC-Light-Diffusion   # IC-Light
```

### Symlink modelli

```bash
# Linka la cartella modelli app → cartella ComfyUI (no duplicati su disco)
ln -s ~/Documents/Vihente\ Forge/models/base    ComfyUI/models/checkpoints
ln -s ~/Documents/Vihente\ Forge/models/lora    ComfyUI/models/loras
ln -s ~/Documents/Vihente\ Forge/models/controlnet  ComfyUI/models/controlnet
ln -s ~/Documents/Vihente\ Forge/models/ipadapter   ComfyUI/models/ipadapter
```

### Avvio app con ComfyUI reale

```bash
python -m src --skip-model-check   # l'app avvia ComfyUI come processo figlio
```

---

## Profili di runtime

```bash
# Sviluppo UI: mock completo (no GPU, no ComfyUI)
python -m src --mock --skip-model-check

# Debug verbose
python -m src --mock --debug

# Runtime reale con ComfyUI
python -m src

# Cartella dati isolata (per test senza toccare dati reali)
python -m src --mock --data-dir ./test_data
```

---

## `requirements.txt` — stack runtime completo

```
# UI
PyQt6==6.7.1

# Image processing
Pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84

# Tagging dataset (ONNX, Fase 2)
onnxruntime-gpu==1.19.2
onnx==1.16.2

# Training (sd-scripts, Fase 2)
# torch + xformers + bitsandbytes installati nel venv sd-scripts separato

# Utility
toml==0.10.2
psutil==6.0.0          # kill orphan processes
websocket-client==1.8.0   # client WebSocket per ComfyUI progress
send2trash==1.8.3      # delete sicuro progetti
watchdog==4.0.2

# Dev (non a runtime)
pytest==8.3.2
pytest-qt==4.4.0
black==24.8.0
ruff==0.6.4
```

**Nota**: `torch`, `diffusers`, `transformers`, `accelerate` NON sono
dipendenze dell'app. Sono dipendenze di ComfyUI (installate nel venv di
ComfyUI separato). L'app non li importa mai.

---

## Training (Fase 2) — sd-scripts separato

Il training LyCORIS usa `sd-scripts` (kohya) in un venv dedicato,
avviato dall'app come subprocess (come ComfyUI).

```bash
# Setup una-tantum (Fase 2)
git clone https://github.com/kohya-ss/sd-scripts.git external/sd-scripts
cd external/sd-scripts
git checkout sdxl                    # branch SDXL
python -m venv .venv-sdscripts
source .venv-sdscripts/bin/activate
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install lycoris_lora             # LyCORIS support
```

---

## Variabili d'ambiente

```bash
# Override cartelle (utile per test isolati)
export VFORGE_APP_DIR=/tmp/vforge-test-app
export VFORGE_DATA_DIR=/tmp/vforge-test-data
export VFORGE_MODELS_DIR=/tmp/vforge-test-models
export VFORGE_PROJECTS_DIR=/tmp/vforge-test-projects

# Modalità offline dopo download iniziale (nessuna connessione rete)
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Riduce frammentazione VRAM (utile 8GB, per ComfyUI)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## Strategia dev

### Branch model
- `main` — release stabili
- `feature/{nome}` — singole feature

### Conventional commits
```
feat(comfy): add WorkflowTemplate with semantic mapping
fix(ui): prevent status bar freeze on GPU poll
refactor(recipes): extract parametrize to pure function
docs(models): add auxiliary model catalog
test(recipe_worker): cover abort signal path
```

### Pre-commit checks
```bash
ruff check src/        # linting
black --check src/     # formatting
pytest tests/ -x       # tutti i test (veloci, no GPU)
```

I test marcati `@pytest.mark.slow` richiedono GPU e ComfyUI reale:
```bash
pytest tests/ -m slow   # solo manualmente
```

---

## Build di release (Fase 7, PyInstaller)

```bash
python scripts/build.py --target windows --version 0.1.0
```

Cosa include il bundle:
1. App PyQt6 + dipendenze leggere
2. ComfyUI pinnato + custom nodes (pre-installati)
3. Script primo avvio (download modelli guidato)

**Non include**: modelli AI (~15-22 GB). Download guidato al primo avvio.
**Dimensione bundle senza modelli**: ~800 MB (PyQt6 + ComfyUI base).

---

## Troubleshooting comune

| Problema | Causa probabile | Fix |
|---|---|---|
| `pytest` fallisce su import PyQt6 | PyQt6 non installato nel venv | `pip install PyQt6` |
| Finestra non compare in `--mock` | Display non disponibile (server headless) | Normale in CI/cloud: verifica la logica con pytest |
| ComfyUI non parte (timeout 60s) | Path ComfyUI sbagliato o dipendenze mancanti | Controlla `logs/comfyui.log`; verifica `engine/ComfyUI/main.py` esiste |
| OOM durante ricetta | Pipeline troppo pesante per 8GB | Aggiungi `--lowvram` in `app_config.comfy_vram_mode`; ComfyUI fa offload automatico |
| Progress WebSocket non arriva | ComfyUI non raggiungibile o client_id mismatch | Verifica `client.is_alive()`; controlla porta in `logs/comfyui.log` |
| ComfyUI orfano alla chiusura | Crash app senza cleanup | Al prossimo avvio `ComfyServer._kill_orphans()` lo termina |
| Output immagine non trovato | File in `comfy_outputs/` ma path sbagliato | Controlla `_fetch_outputs` in `client.py` |
| LoRA non applicato | `project.active_lora` è None o path non trovato | Verifica che il progetto abbia un training run completato |
| `send2trash` non installato | Dipendenza opzionale mancante | `pip install send2trash`; fallback rinomina file |

---

## Convenzioni di codice

- **`from __future__ import annotations`** in cima a ogni file Python
- **Type hints** su tutte le funzioni pubbliche
- **Solo `pathlib.Path`** per i path, mai `os.path` o stringhe raw
- **`logger = logging.getLogger(__name__)`** per modulo, mai `print()`
- **Qt signals**: `something_happened` / slot `_on_something_happened`
- **Threading**: mai mutare widget Qt fuori dal main thread — sempre signal
- **Test**: `conftest.py` isola le path utente in `tmp_path` via env var

---

## Risorse esterne

- ComfyUI repo: https://github.com/comfyanonymous/ComfyUI
- ComfyUI API reference: `ComfyUI/script_examples/` nel repo
- PyQt6 docs: https://doc.qt.io/qtforpython-6/
- sd-scripts (training): https://github.com/kohya-ss/sd-scripts/wiki
- LyCORIS: https://github.com/KohakuBlueleaf/LyCORIS
- Pony V6 XL prompting guide: https://civitai.com/articles/3848
- IP-Adapter FaceID: https://github.com/tencent-ailab/IP-Adapter
