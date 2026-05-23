# Studio Mode

Modalità alternativa, separata da Forge Mode. Generazione libera stile
Tensor Art / A1111: scegli modello base + LoRA scaricati da te + prompt
→ immagine. Per usare modelli e LoRA di terzi (Civitai, HuggingFace).

**Stato: predisposto in architettura, sviluppo dopo Forge Mode base.**

## Decisioni fissate

| Aspetto | Decisione | Implicazione |
|---|---|---|
| Relazione con Forge | **Separazione totale** | Studio NON legge i progetti Forge. I LoRA addestrati restano privati di Forge. |
| Download modelli | **Manuale** | L'utente scarica e mette i file in cartelle. L'app non fa networking verso Civitai/terzi. |
| Priorità sviluppo | **Dopo Forge base** | Architettura predisposta ora, UI sviluppata in fase successiva. |

## Perché due modalità separate

Hanno filosofie opposte e mescolarle confonderebbe entrambe:

- **Forge Mode**: fedeltà al tuo stile, complessità nascosta, tutto
  guidato, le ricette (personaggio in posa, prodotto, correzione).
  Contenuto = tuo, creato da te.
- **Studio Mode**: controllo totale, sperimentazione, l'utente sa cosa
  fa. Solo generazione libera. Contenuto = roba del mondo, scaricata.

Condividono il motore (ComfyUI), il download manager interno, la
gestione VRAM. Studio Mode è UI nuova sopra infrastruttura comune, non
un secondo programma.

## Il launcher

All'avvio l'app mostra una scelta (saltabile con "ricorda scelta"):

```
┌─────────────────────────────────────────────┐
│            Vihente Forge                      │
│   Cosa vuoi fare oggi?                         │
│   ┌──────────────────┐  ┌──────────────────┐ │
│   │   FORGE MODE      │  │   STUDIO MODE     │ │
│   │  Il tuo stile,    │  │  Modelli e LoRA   │ │
│   │  progetti,        │  │  liberi, stile    │ │
│   │  ricette          │  │  Tensor Art       │ │
│   │  [ Entra ]        │  │  [ Entra ]        │ │
│   └──────────────────┘  └──────────────────┘ │
│   [ ] Ricorda la scelta                        │
└─────────────────────────────────────────────┘
```

La preferenza si salva in config app. Si può sempre tornare al launcher
dal menu di ciascuna modalità.

## Cartelle Studio Mode

Completamente separate da quelle Forge:

```
~/Documents/Vihente Forge/studio/
├── checkpoints/     ← modelli base (.safetensors) scaricati dall'utente
├── loras/           ← LoRA scaricati
├── embeddings/      ← textual inversion (opzionale)
└── vae/             ← VAE custom (opzionale)
```

L'utente mette un file, l'app lo rileva al refresh (o riavvio), appare
nel menu. Nessun download automatico, nessuna scansione delle cartelle
Forge.

## UI Studio Mode (futura)

```
┌──────────────────────────────────────────────┐
│  Studio Mode                    [← Launcher]   │
├──────────────────────────────────────────────┤
│  Modello base:  [ scegli dai tuoi ▼ ]  [badge]│
│  LoRA:          [ + Aggiungi ]                 │
│    • lora_a    peso [0.8] [×]                  │
│  Prompt:    [____________________________]     │
│  Negative:  [____________________________]     │
│  Steps[30] CFG[7] Size[1024²] Seed[-1]         │
│  Sampler [DPM++ 2M Karras ▼]                   │
│              [    Genera    ]                  │
└──────────────────────────────────────────────┘
```

## Gestione dell'incoerenza modello/LoRA

Il limite tecnico chiave: **un LoRA funziona solo con la famiglia del
modello base su cui è stato addestrato** (SD1.5 / SDXL / Pony / FLUX /
Illustrious). Accoppiate sbagliate = artefatti o nessun effetto.

Mitigazioni in Studio Mode:

1. **Detector famiglia**: legge l'header safetensors → identifica la
   famiglia → mostra badge ("SDXL", "SD 1.5", "FLUX"...).
2. **Filtro compatibilità**: mostra solo i LoRA compatibili col base
   scelto (come fa Tensor Art).
3. **Lettura metadata LoRA**: molti LoRA Civitai hanno trigger words e
   modello consigliato nei metadata. Se presenti, mostrati all'utente.
4. **Warning pre-generazione**: se l'accoppiata è incompatibile, avvisa
   prima di sprecare tempo di calcolo.
5. **Nessuna garanzia di qualità**: dichiarato nell'UI. Studio usa roba
   di terzi, la qualità dipende da quella.

## Cosa Studio Mode NON fa

- **Niente ricette complesse.** Personaggio-in-posa, prodotto-in-scena,
  correzione restano esclusiva di Forge Mode, perché tarate su stack
  SDXL/Pony specifico. Studio fa solo txt2img e img2img base.
- **Niente training.** Studio usa modelli esistenti, non ne addestra.
- **Niente accesso ai progetti Forge.** Separazione totale.
- **Niente download automatico.** L'utente porta i file.

Se in futuro si volessero le ricette anche in Studio, servirebbe gestire
varianti dei modelli ausiliari per ogni famiglia (ControlNet SD1.5 ≠
ControlNet SDXL, ecc). Grande complessità, rimandata indefinitamente.

## Detector famiglia modello (nota tecnica)

L'identificazione della famiglia da un .safetensors si basa su:
- Dimensioni dei tensori chiave (es. cross-attention dim: SD1.5=768,
  SDXL=2048)
- Presenza/assenza di specifiche chiavi nei pesi
- Metadata header se presenti (`ss_base_model_version` nei LoRA kohya)

Implementato in `src/studio/model_scanner.py` (predisposto, vedi
scaffold). Non richiede caricare il modello in VRAM: legge solo l'header
del safetensors (veloce, pochi KB).
