# Architettura tecnica

## Vista d'insieme

```
┌───────────────────────────────────────────────────────────┐
│                       PyQt6 UI (main thread)               │
│  ┌────────────┬──────────────────────────────────────┐    │
│  │ Sidebar    │ Workspace (tab per progetto attivo)  │    │
│  │ progetti   │ ├─ Dataset (griglia + tag editor)    │    │
│  │            │ ├─ Train  (preset + progress)        │    │
│  │            │ ├─ Generate (prompt + preview)       │    │
│  │            │ └─ Gallery (output promossi)         │    │
│  └────────────┴──────────────────────────────────────┘    │
└────────────────────┬──────────────────────────────────────┘
                     │ Qt signals (thread-safe)
        ┌────────────┼────────────┬─────────────┐
        ▼            ▼            ▼             ▼
   Generation   Training     Tagging      Upscale
   Worker       Worker       Worker       Worker
   (QThread)    (subprocess) (QThread)    (QThread)
        │            │            │             │
        ▼            ▼            ▼             ▼
   diffusers    sd-scripts    WD-tagger    Real-ESRGAN
   pipeline     (kohya)       ONNX         model
        │            │            │             │
        └────────────┴─────┬──────┴─────────────┘
                           ▼
                    PyTorch + CUDA
                    Modelli su disco
```

## Perché PyQt6 e non altro

- **Coerenza**: Bru ha già RCS-App in PyQt5. Passaggio a PyQt6 minimo, sintassi quasi identica.
- **Performance grafica**: per visualizzare 100+ thumbnail in griglia con zoom fluido serve Qt nativo, non Electron.
- **Threading reale**: QThread + signal-slot è il modo più pulito di gestire generazione lunga senza freeze UI.
- **Distribuzione**: PyInstaller con PyQt6 è collaudato su Windows.

PyQt6 vs PySide6: equivalenti tecnicamente. Si usa PyQt6 perché ha license GPL/commercial chiara e community più grande per troubleshooting; se in futuro serve distribuzione commerciale closed-source si passa a PySide6 (LGPL) con cambio di import minimo.

## Thread model

**Mai bloccare il main thread.** Tutto ciò che dura più di 100ms va su worker.

| Operazione | Tipo worker | Durata tipica |
|---|---|---|
| Generazione immagine | QThread con pipeline diffusers | 10-30s |
| Training LoRA | subprocess (sd-scripts) | 1-4 ore |
| Auto-tagging dataset | QThread con WD-tagger ONNX | 1-5 min |
| Upscale 4x | QThread Real-ESRGAN | 5-15s |
| Download modello | QThread huggingface_hub | 5-30 min |

**Training è subprocess, non QThread.** Motivo: `sd-scripts` è uno script Python complesso che usa accelerate, gestione memoria custom, e a volte va terminato hard. Tenerlo in subprocess separato ha tre vantaggi:
1. Se crasha, la UI sopravvive
2. Si può uccidere pulito con SIGTERM senza killare l'app
3. La memoria torch del training viene rilasciata totalmente alla fine (importante: training + generazione nello stesso processo Python possono lasciare frammentazione VRAM)

La comunicazione subprocess → UI passa per parsing del log su stdout, gestito da `training/monitor.py`.

## Gestione VRAM su 8GB

**Generazione** (SDXL fp16):
```python
pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()      # sposta componenti in RAM quando inattivi
pipe.enable_vae_tiling()             # VAE decode in chunk per output >1024
# Il VAE fp16 default ha bug → si sostituisce con sdxl-vae-fp16-fix
```
Picco VRAM ~6.5GB per 1024×1024. Tempo: 12-15s su RTX 3060.

**Training LoRA SDXL** (sd-scripts):
```
--mixed_precision bf16
--gradient_checkpointing
--xformers              (riduce attention VRAM 40%)
--cache_latents         (precalcola VAE encode, evita ricalcoli)
--cache_text_encoder_outputs  (idem text encoder, libera 1.5GB durante training)
--network_dim 16        (rank LoRA basso, default standard è 32)
--network_alpha 8
--train_batch_size 1
--gradient_accumulation_steps 4  (simula batch 4 con VRAM di batch 1)
--optimizer_type AdamW8bit  (bitsandbytes, dimezza VRAM optimizer)
--max_data_loader_n_workers 0  (su Windows, multiprocessing data loader instabile)
```
Picco VRAM ~7.5GB. Tempo: ~2h per 100 immagini, 10 epoch, SDXL.

Se OOM persistente: si scende a SD 1.5 base, training in 30-45 min, qualità leggermente inferiore ma comunque buona per la maggior parte dei casi.

**Training e generazione non vanno mai eseguiti in parallelo.** L'UI disabilita la sezione Generate quando un training è in corso (e viceversa).

## Auto-tagging: come funziona

Il tagger non capisce "lo stile di Bru". Capisce **etichette concrete** ("1girl, dark fantasy, red eyes, crow, gothic dress, full moon, cemetery"). Sono queste etichette + il **tag attivatore unico del progetto** (es. `vfstyle_iris`) che insegnano allo stile.

Pipeline:
1. WD-tagger ONNX (modello ~400MB) processa ogni immagine → restituisce lista tag con confidence
2. Si filtrano tag sotto soglia (default 0.35)
3. Si aggiunge in cima il tag attivatore del progetto
4. Si genera `image.txt` accanto a `image.png`

L'utente può poi:
- Vedere i tag in griglia con UI rapida (tab/enter per scorrere immagini)
- Aggiungere tag personalizzati (es. "vihente_iris_v2" per varianti)
- Rimuovere tag errati (es. WD ha detto "school uniform" su un'immagine che ha solo una giacca)
- Operazioni bulk (rimuovi tag X da tutte, aggiungi tag Y a selezione)

**Importanza del tag attivatore.** È la "parola magica" che attiva lo stile in fase di generazione. Va univoco e non collidente con tag esistenti del modello base. Convenzione: `vf_{progetto}_{versione}`, es. `vf_iris_v1`. L'UI lo genera automaticamente e lo mostra read-only in lettura, modificabile solo in advanced.

## Modello base: scelta e implicazioni

Il modello base è il "talento generale" su cui si costruisce lo stile specifico. Non lo scegli ogni volta che generi: lo scegli **alla creazione del progetto** e resta fisso per la vita del progetto.

Cambiarlo dopo il training **invalida il LoRA addestrato**: l'UI mostra
warning e propone un re-training automatico.

Modelli candidati pre-installati al primo avvio dell'app:

| Modello base | Dominio | Quando sceglierlo |
|---|---|---|
| Pony Diffusion V6 XL | Character art, illustrazione anime/cartoon | Iris, sprite stilizzati, character design |
| Juggernaut XL v9 | Fotorealismo, prodotti | Render carbon fiber, foto-realistic mockup |
| AnimagineXL 4.0 | Anime/illustration alta qualità | Alternativa a Pony, output più "pulito" |
| SD 1.5 (RealisticVision) | Generico veloce, retrocompatibilità | Sprite pixel, training rapido, esperimenti |

L'utente non vede questa lista in modo opzionistico ostile. Vede una scelta guidata: "Che tipo di lavoro vuoi addestrare?" → l'app suggerisce il base appropriato, ma è cambiabile in advanced.

## Persistenza output

Ogni immagine generata produce:
1. **PNG con metadata embedded** nel chunk tEXt (formato compatibile A1111):
   ```
   parameters: <prompt>
   Negative prompt: <negative>
   Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7.5,
   Seed: 1234567890, Size: 1024x1024,
   Model: pony-v6-xl, Lora: vf_iris_v1:0.8,
   App: Vihente Forge v0.1.0
   ```
2. **JSON sidecar** completo (stesso nome, .json) con struttura dataclass `GenerationResult`

Vantaggi:
- Drop di una vecchia PNG sull'app ricarica tutti i parametri ("rigenera con questi settings")
- Portabilità verso A1111/ComfyUI per power user
- Storia completa senza database

## Sicurezza e isolamento

- `HF_HUB_OFFLINE=1` dopo il primo download (rifiuta connessioni rete)
- `HF_HUB_DISABLE_TELEMETRY=1` sempre
- `safetensors` only per modelli scaricati (no pickle eseguibile)
- I prompt vanno solo a log locale, mai a servizio remoto
- Nessun crash reporter automatico (eventuali log restano locali)

## Strategia di test

Test automatici limitati (la maggior parte del valore è in interazione utente), ma:

- **Unit**: `utils/`, `core/lora_loader.py`, parser tag, paths
- **Integration**: `core/project.py` (creazione/load/save progetto su tmpdir)
- **Smoke**: lancio app headless con `QT_QPA_PLATFORM=offscreen`, generazione single image con modello fake, verifica file output

Test di training NON automatici (durerebbero ore, richiedono GPU). Si fa test manuale su dataset noto ridotto (10 immagini, 200 step) prima di ogni release.

## Estensibilità futura (non Fase 1)

L'architettura prevede questi hook senza implementarli subito:

- **Plugin system**: cartella `~/.vihente-forge/plugins/`, Python files con interfaccia definita per aggiungere processori custom (es. un post-processor che applica un effetto specifico)
- **API locale**: server FastAPI opzionale (porta 7860) per integrazione con altri tool — utile se Bru vuole automatizzare batch da script
- **Multi-LoRA stacking**: combinare 2-3 LoRA propri ("stile Iris" + "stile fondali" → personaggio Iris in ambientazione)
- **Video frames**: estensione AnimateDiff per generare brevi animazioni nello stile appreso

## Decisioni esplicite di NON fare

- **No cloud sync.** Mai. Anche se l'utente lo chiede esplicitamente. Sicurezza by default.
- **No prompt enhancement via LLM cloud.** Se in futuro si vuole "migliora questo prompt", si usa un LLM locale (Llama 3 / Qwen) opzionale, mai API esterne.
- **No ComfyUI embedded.** Resta strumento di power user esterno.
- **No collaboration/sharing built-in.** Lo scopo è personale. Export di un progetto è una cartella zip, l'utente decide cosa farne.
- **No store di LoRA community.** L'app non scarica LoRA di terzi automaticamente. Si possono importare manualmente da Civitai etc, ma con warning chiaro.
