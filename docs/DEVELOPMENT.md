# Sviluppo

## Requisiti hardware (sviluppo + runtime)

- **GPU**: NVIDIA con 8+ GB VRAM, compute capability 7.0+ (RTX 2060+, RTX 30xx/40xx)
- **CUDA Toolkit**: 12.1 (compatibile con torch 2.4)
- **Driver NVIDIA**: 535+ (Windows) / 535+ (Linux)
- **RAM sistema**: 16 GB minimo, 32 GB consigliato (per cache latents + UI fluida)
- **Disco**: 50 GB liberi (10 venv + 30 modelli + spazio progetti)
- **OS**: Windows 10/11, Linux (Ubuntu 22.04+), macOS (solo CPU per ora, no training)

## Requisiti software

- **Python 3.11.x** — NON 3.12 (torch wheel + bitsandbytes hanno problemi su Windows con 3.12)
- **Git** per clonare sd-scripts
- **Visual Studio Build Tools 2022** su Windows (per compilare alcune dipendenze native, es. xformers se non c'è wheel pre-built)

## Setup Windows

```powershell
# 1. Verifica Python 3.11
python --version
# Atteso: Python 3.11.x

# 2. Verifica CUDA
nvidia-smi
# Atteso: CUDA Version: 12.1 o superiore

# 3. Clone repo (futuro)
git clone <repo> C:\dev\vihente-forge
cd C:\dev\vihente-forge

# 4. Crea virtual environment
python -m venv .venv
.venv\Scripts\activate

# 5. Aggiorna pip + setuptools
python -m pip install --upgrade pip setuptools wheel

# 6. Installa PyTorch CUDA 12.1 PRIMA del resto (versione esatta)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 7. Resto delle dipendenze
pip install -r requirements.txt

# 8. Clone sd-scripts (engine training)
git clone https://github.com/kohya-ss/sd-scripts.git external/sd-scripts
cd external\sd-scripts
git checkout sdxl
pip install --no-deps -r requirements.txt
cd ..\..

# 9. Verifica torch + CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
# Atteso: CUDA: True, GPU name, VRAM > 7 GB

# 10. Avvia app in dev mode
python -m src
```

## Setup Linux (Ubuntu 22.04+)

```bash
# 1. Dipendenze sistema
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev git build-essential

# 2. CUDA Toolkit (se non già presente)
# Segui guida NVIDIA per CUDA 12.1
# Verifica:
nvidia-smi
nvcc --version

# 3. Clone repo
git clone <repo> ~/dev/vihente-forge
cd ~/dev/vihente-forge

# 4. Venv
python3.11 -m venv .venv
source .venv/bin/activate

# 5. Pip upgrade
pip install --upgrade pip setuptools wheel

# 6. PyTorch
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 7. Dipendenze
pip install -r requirements.txt

# 8. sd-scripts
git clone https://github.com/kohya-ss/sd-scripts.git external/sd-scripts
(cd external/sd-scripts && git checkout sdxl && pip install --no-deps -r requirements.txt)

# 9. Verifica
python -c "import torch; print(torch.cuda.is_available())"

# 10. Run
python -m src
```

## `requirements.txt` (riferimento)

```
# UI
PyQt6==6.7.1
PyQt6-Qt6==6.7.2

# Inference engine
diffusers==0.30.3
transformers==4.44.2
accelerate==0.33.0
safetensors==0.4.5
huggingface-hub==0.24.6

# Image processing
Pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84

# Tagging (ONNX)
onnxruntime-gpu==1.19.2
onnx==1.16.2

# Training support (sd-scripts uses these too)
xformers==0.0.27.post2
bitsandbytes==0.43.3        # Windows: usare bitsandbytes-windows-webui se quella ufficiale dà problemi
prodigyopt==1.0
lion-pytorch==0.2.2

# Upscaler
realesrgan==0.3.0
basicsr==1.4.2

# Util
toml==0.10.2
psutil==6.0.0
watchdog==4.0.2

# Dev (opzionale)
pytest==8.3.2
pytest-qt==4.4.0
black==24.8.0
ruff==0.6.4
```

## Variabili d'ambiente importanti

```bash
# Disabilita telemetria HuggingFace
export HF_HUB_DISABLE_TELEMETRY=1

# Cartella modelli custom (default: ~/Documents/Vihente Forge/models)
export VFORGE_MODELS_DIR=/path/custom

# Modalità offline (post-download iniziale)
export HF_HUB_OFFLINE=1

# Riduce frammentazione VRAM (utile su 8GB)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Su Windows con xformers, può servire:
export XFORMERS_FORCE_DISABLE_TRITON=1
```

L'app setta queste automaticamente in `__main__.py` PRIMA di importare
torch/diffusers (importante: l'ordine conta).

## Strategia dev

### Branch model
- `main` — solo release stabili
- `develop` — integration branch
- `feature/{nome}` — sviluppo singole feature

### Conventional commits (suggerito)
```
feat(training): add resume from checkpoint
fix(ui): prevent freeze during dataset import
refactor(core): extract LoRA loader to module
docs(readme): update setup instructions
```

### Pre-commit checks
- `ruff check src/` — linting
- `black --check src/` — formatting
- `pytest tests/ -m "not slow"` — fast tests

Le slow test (integration con GPU reale) si lanciano solo manualmente:
```bash
pytest tests/ -m slow
```

## Profili di runtime

L'app supporta flag CLI per development:

```bash
# Modalità normale
python -m src

# Mock pipeline (no GPU, per test UI rapidi)
python -m src --mock

# Log verbose
python -m src --debug

# Test progetto isolato
python -m src --data-dir ./test_data

# Skip download check (se modelli già presenti)
python -m src --skip-model-check
```

## Build di release (PyInstaller)

Lo script `scripts/build.py` gestisce build pulita. Riferimento:

```bash
python scripts/build.py --target windows --version 0.1.0
```

Cosa fa:
1. Clean `build/`, `dist/`
2. Verifica dipendenze
3. PyInstaller one-folder con `--collect-all PyQt6 --collect-all diffusers`
4. Strip simboli debug
5. Copia README, LICENSE, asset
6. Crea installer NSIS se Windows
7. Zip finale in `dist/vihente-forge-{version}-{platform}.zip`

**Dimensione attesa build Windows**: ~3 GB (PyTorch + CUDA libs è grosso).
Modelli scaricati separatamente al primo avvio (NON inclusi nel bundle).

## Troubleshooting comune

| Problema | Causa probabile | Fix |
|---|---|---|
| `torch.cuda.is_available()` False | Driver NVIDIA o CUDA non installato | Reinstalla driver, riavvia. Verifica `nvidia-smi` |
| OOM al primo generate | Altre app usano VRAM | Chiudi Chrome/giochi. Verifica `nvidia-smi` mostra <500MB usati |
| `xformers` import error | Versione mismatch con torch | `pip install xformers --no-deps` e prendi la versione che corrisponde al torch |
| Training crash immediato | sd-scripts non installato bene | Verifica `external/sd-scripts/` esista e i suoi requirements installati |
| UI freeze durante generation | Worker non in QThread | Verifica decorator `@worker_thread` o subclass corretta |
| Output sempre nero | VAE bug fp16 SDXL | Forza `madebyollin/sdxl-vae-fp16-fix` o `no_half_vae = true` per training |
| `bitsandbytes` error su Windows | Wheel ufficiale a volte rotta | Usa `bitsandbytes-windows-webui` come fallback |
| Hugging Face timeout | Rate limit o connessione lenta | Aumenta `HF_HUB_DOWNLOAD_TIMEOUT=60` |

## Git ignore essenziale

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/

# Asset/dati locali
.vforge_data/
external/sd-scripts/
build/
dist/
*.spec
*.log

# IDE
.vscode/
.idea/
*.swp
```

## Convenzioni di codice

- **Type hints sempre** su funzioni pubbliche
- **Docstring** stile Google su classi e metodi pubblici (non su tutto)
- **`from __future__ import annotations`** in cima a ogni file Python
- **Path: solo `pathlib.Path`**, mai `os.path` o stringhe
- **Logger named** per modulo: `logger = logging.getLogger(__name__)`, mai `print()`
- **Qt naming**: signal `something_happened`, slot `_on_something_happened`
- **Threading**: nessuna mutazione di Qt widget fuori dal main thread, sempre signal

## Risorse esterne utili

- diffusers docs: https://huggingface.co/docs/diffusers
- sd-scripts wiki: https://github.com/kohya-ss/sd-scripts/wiki
- PyQt6 docs: https://doc.qt.io/qtforpython-6/
- Pony V6 XL prompting guide: https://civitai.com/articles/3848
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
