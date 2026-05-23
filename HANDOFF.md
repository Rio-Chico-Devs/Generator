# Vihente Forge — Handoff

Assistente grafico personale, offline e indipendente, che apprende lo
stile di Bru e lo applica a compiti reali da graphic designer: mettere
un personaggio in una posa scelta, comporre un volantino attorno a un
prodotto, correggere e rifinire immagini senza inventare elementi
estranei.

Non è un generatore generico. Non emula Midjourney né altri. È uno
strumento privato addestrato da noi, per noi, su materiale nostro.

---

## I quattro principi non negoziabili

**1. Indipendenza totale da fonti esterne.**
Dopo l'installazione iniziale, l'app gira completamente offline e
NON dipende da alcun servizio remoto in runtime. Nessuna API cloud,
nessun account, nessun check di licenza online, nessuna telemetria.
ComfyUI, i modelli, i LoRA, i workflow: tutto vive sul disco di Bru.
Se internet sparisce, l'app continua identica. I modelli si possono
anche copiare da chiavetta per installazione 100% offline.

**2. Addestrato da noi, per noi.**
Lo scopo non è "fare cose belle in generale". È replicare fedelmente
lo stile di Bru e svolgere i compiti specifici di Bru. Ogni scelta
tecnica privilegia la fedeltà al materiale fornito sopra la
versatilità generica.

**3. Lavoro da grafico, fatto bene.**
L'obiettivo di qualità è "output usabile in produzione", non "demo
carina". Questo richiede pipeline composite multi-step (non un singolo
prompt), con correzione automatica dei difetti tipici (mani, volti,
anatomia) e controllo preciso su posa, composizione e identità del
personaggio.

**4. Semplicità per l'utente, complessità nascosta.**
Bru non deve sapere cos'è un ControlNet o un IP-Adapter. Vede "ricette"
semplici: "Personaggio in posa", "Prodotto in scena", "Correggi
immagine". Sotto, ognuna orchestra 5-8 modelli. La complessità è reale
ma invisibile.

---

## Due modalità: Forge e Studio

L'app ha un **launcher** all'avvio con due modalità separate.

**Forge Mode** (il cuore, priorità di sviluppo): i tuoi progetti, lo
stile addestrato da te, le ricette (personaggio in posa, prodotto in
scena, correzione), la pose library. Contenuto creato da te, cresce nel
tempo, è la tua estensione creativa.

**Studio Mode** (extra, sviluppo dopo Forge): generazione libera stile
Tensor Art / A1111. Scegli un modello base + LoRA che hai scaricato tu
(Civitai, ecc.) + prompt → immagine. Per sperimentare con roba di terzi.

Decisioni fissate su Studio (vedi docs/STUDIO_MODE.md):
- **Separazione totale**: Studio NON tocca i progetti Forge. I tuoi
  stili addestrati restano privati di Forge.
- **Download manuale**: metti i file in cartelle, nessun networking
  automatico verso Civitai. Indipendenza in runtime preservata.
- **Niente ricette in Studio**: solo txt2img/img2img base. Le ricette
  complesse sono esclusiva Forge (stack SDXL/Pony controllato).
- **Sviluppo dopo Forge base**: architettura predisposta ora (launcher,
  model scanner con detector famiglia), UI completa dopo.

Le due modalità condividono il motore ComfyUI, il download manager e la
gestione VRAM. Studio è UI nuova sopra infrastruttura comune.

---

## Cosa fa concretamente — le tre ricette

### Ricetta A — Personaggio in posa  (PRIORITÀ 1)
Bru fornisce: una posa (da Pose Library o foto/sketch caricato al
momento) + un'immagine del personaggio + stile (LoRA addestrato).
L'app produce: il personaggio di Bru, nella posa esatta, nel suo stile.

Pipeline interna:
```
posa reference → estrazione scheletro (OpenPose)
              → ControlNet OpenPose (vincola la posa)
personaggio   → IP-Adapter FaceID (mantiene identità)
stile         → LoRA Bru (linguaggio visivo)
generazione   → output grezzo
              → Adetailer (auto-fix volto + mani)
              → upscale finale opzionale
```

### Ricetta B — Prodotto in scena  (PRIORITÀ 2, focus RCS)
Bru fornisce: foto di un prodotto reale (da una pool/libreria prodotti)
+ indicazione del tipo di output (volantino, post social, hero banner).
L'app produce: composizione grafica nello stile di Bru, con il prodotto
integrato in modo coerente (luce, ombra, prospettiva), spazio per testo.

Pipeline interna:
```
foto prodotto → background removal (BiRefNet)
             → scena/background generato nel tuo stile (con spazio prodotto)
             → re-illuminazione coerente del prodotto (IC-Light)
             → compositing prodotto + scena
             → layer testo come SVG overlay (font reali, non AI text)
             → final pass uniformante
```

### Ricetta C — Correggi / rifinisci  (PRIORITÀ 3, trasversale)
Bru fornisce: un'immagine + selezione zona da correggere (brush).
L'app produce: la stessa immagine con la zona corretta, SENZA toccare
il resto e SENZA aggiungere elementi non richiesti.

Pipeline interna:
```
immagine → selezione mask (brush UI)
        → ControlNet Reference (preserva il resto al 100%)
        → inpainting solo nella mask, strength bassa (0.4-0.6)
        → multiple seed in parallelo → utente sceglie
```

### Più la base: Genera nel mio stile
Il caso semplice prompt → immagine nel tuo stile resta disponibile,
come punto di partenza per esplorazione.

---

## Stack tecnico (definitivo)

| Layer | Scelta | Perché |
|---|---|---|
| GUI | PyQt6 | Coerenza RCS-App, controllo, performance |
| Engine generazione | **ComfyUI headless** (embedded, API mode) | Unico modo di orchestrare pipeline multi-step in modo robusto e manutenibile. Gira in locale, offline. |
| Engine training | sd-scripts (kohya) + LyCORIS | Massima fedeltà stile (vedi sotto) |
| Captioning dataset | Llama 3.2 Vision (locale, opzionale) | Caption fedeli al concetto, offline |
| Modello base | Pony Diffusion V6 XL | Eccellente character, tag Danbooru (workflow Bru), SDXL |
| Pose extraction | DWPose / OpenPose | Estrae scheletro da reference |
| Character identity | IP-Adapter FaceID Plus v2 | Mantiene personaggio riconoscibile |
| Background removal | BiRefNet | Sostituto open di remove.bg, locale |
| Re-illuminazione | IC-Light | Integra prodotti in scene coerenti |
| Auto-fix difetti | Adetailer (face + hand) | Risolve mani/volti distorti automaticamente |
| Upscale | Real-ESRGAN / 4x-UltraSharp | Output hi-res produzione |
| Distribuzione | PyInstaller + ComfyUI bundle | Windows primario |

**Perché LyCORIS e non LoRA standard per il training:**
LoRA standard modifica solo i layer di attention. LyCORIS modifica
anche i layer convoluzionali, dove vivono texture, tratto e pennellata
— esattamente ciò che distingue lo stile di Bru dal generico. Fedeltà
attesa 90-93% contro 75-80% del LoRA standard. Stesso costo di training.

---

## Architettura a due processi

```
┌──────────────────────────────────────────────┐
│  Vihente Forge — PyQt6 (processo principale)  │
│  ┌──────────┬─────────────────────────────┐  │
│  │ Sidebar  │ Workspace                   │  │
│  │ progetti │  - Genera / Ricette         │  │
│  │ + Pose   │  - Dataset & Training       │  │
│  │ Library  │  - Galleria                 │  │
│  └──────────┴─────────────────────────────┘  │
└────────────────────┬──────────────────────────┘
                     │ HTTP localhost (127.0.0.1:8188)
                     │ + WebSocket per progress
┌────────────────────▼──────────────────────────┐
│  ComfyUI headless (processo figlio)            │
│  - Avviato dall'app, ucciso alla chiusura      │
│  - Carica workflow JSON da assets/workflows/   │
│  - Solo localhost, mai esposto in rete         │
└────────────────────┬──────────────────────────┘
                     │
┌────────────────────▼──────────────────────────┐
│  Modelli locali (base, LoRA, ControlNet,       │
│  IP-Adapter, upscaler) in ~/Documents/...      │
└────────────────────────────────────────────────┘
```

ComfyUI gira come server locale che SOLO la nostra app conosce
(bind 127.0.0.1, mai 0.0.0.0). L'utente non lo vede e non lo apre mai.
Le "ricette" sono file workflow JSON in `assets/workflows/`, caricati e
parametrizzati dalla nostra UI prima dell'invio.

---

## I due tipi di "memoria" del sistema

Distinzione fondamentale, da non confondere:

**1. Stile (LoRA/LyCORIS addestrato)** — il "come disegna Bru".
File ~150-400MB per progetto. Si addestra sui lavori FINITI di Bru
(50-200 immagini curate). È il linguaggio visivo.

**2. Pose Library** — il "catalogo di pose pronte".
Le ~500 reference di Bru (foto, sketch, qualunque cosa). NON servono per
il training, servono come input ControlNet al volo. Lo stile delle
reference è irrilevante: conta solo lo scheletro/posa che se ne estrae.
Organizzate, categorizzate, ricercabili (vedi docs/POSE_LIBRARY.md).

Quando Bru crea "personaggio in posa": sceglie una posa dalla Library
(o ne carica una nuova), l'app estrae lo scheletro, applica il LoRA
stile, mantiene l'identità del personaggio via IP-Adapter. Tre memorie
diverse che collaborano.

---

## Roadmap a fasi

### Fase 0 — Fondamenta engine (settimane 1-2)
- Scaffold PyQt6 (già fatto, da estendere)
- Embedding ComfyUI headless: avvio/arresto, health check, gestione porta
- Client Python per ComfyUI API (submit workflow, poll progress, retrieve)
- Sistema progetti (già fatto)
- Modalità mock per sviluppo UI senza GPU

### Fase 1 — Base + Ricetta A parte 1 (settimane 3-4)
- Download manager modelli (base + ControlNet OpenPose + IP-Adapter + upscaler)
- Workflow "Genera nel mio stile" (base txt2img con LoRA)
- Pose Library: import, categorizzazione, ricerca, preview
- Estrazione posa da reference (DWPose)

### Fase 2 — Training ad alta fedeltà (settimane 5-6)
- Dataset management (import, processing, manifest)
- Auto-tagging WD + caption Llama Vision opzionale
- Training LyCORIS con preset ottimizzati 8GB
- Benchmark visivo (sample dai prompt reali del dataset)
- Checkpoint selection

### Fase 3 — Ricetta A completa (settimana 7)
- Workflow "Personaggio in posa" end-to-end
- ControlNet OpenPose + IP-Adapter FaceID + LoRA in pipeline
- Adetailer auto-fix volto/mani
- UI ricetta semplice (scegli posa, scegli personaggio, genera)

### Fase 4 — Ricetta B prodotto (settimane 8-9)
- Pool prodotti (libreria immagini prodotto)
- Background removal BiRefNet
- IC-Light re-illuminazione
- Compositing + layer testo SVG
- Template volantino/social/banner

### Fase 5 — Ricetta C + rifinitura (settimana 10)
- Inpainting canvas con brush
- ControlNet Reference per preservazione
- Workflow correzione mirata
- Refinement loop (re-inpaint zone deboli)

### Fase 6 — Power & vettoriali (settimane 11-12)
- Vectorize (vtracer) per output logo/icone → SVG
- LoRA stacking (combina stili)
- Loop apprendimento (gallery → re-training)
- Batch processing

### Fase 7 — Distribuzione
- PyInstaller + ComfyUI bundle Windows
- Installer, prima-esecuzione, documentazione utente

---

## Struttura repository (aggiornata)

```
vihente-forge/
├── HANDOFF.md                    ← questo
├── README.md
├── docs/
│   ├── ARCHITECTURE.md           ← engine, processi, VRAM
│   ├── COMFY_ENGINE.md           ← embedding ComfyUI, API, lifecycle
│   ├── RECIPES.md                ← le 3 pipeline spiegate nel dettaglio
│   ├── POSE_LIBRARY.md           ← gestione 500 reference
│   ├── PROJECTS.md               ← anatomia progetto
│   ├── TRAINING.md               ← LyCORIS, preset, fedeltà
│   ├── MODELS.md                 ← catalogo completo (base + ausiliari)
│   └── DEVELOPMENT.md            ← setup ambiente
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __main__.py
│   ├── app.py
│   ├── comfy/                    ← NUOVO: engine ComfyUI
│   │   ├── server.py             ← avvio/arresto processo ComfyUI
│   │   ├── client.py             ← client API (submit, progress, fetch)
│   │   └── workflow.py           ← carica + parametrizza workflow JSON
│   ├── pose/                     ← NUOVO: pose library
│   │   ├── library.py            ← gestione reference, categorie, ricerca
│   │   └── extractor.py          ← estrazione scheletro DWPose
│   ├── core/
│   │   ├── project.py
│   │   ├── recipes.py            ← NUOVO: definizione ricette A/B/C
│   │   ├── model_manager.py
│   │   ├── catalog.py            ← esteso con modelli ausiliari
│   │   └── config.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── dataset_prep.py
│   │   ├── presets.py            ← preset LyCORIS
│   │   └── monitor.py
│   ├── workers/
│   │   ├── recipe_worker.py      ← NUOVO: esegue ricette via ComfyUI
│   │   ├── training_worker.py
│   │   └── download_worker.py
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── recipe_views/         ← NUOVO: una view per ricetta
│   │   ├── pose_library_view.py  ← NUOVO
│   │   ├── dataset_view.py
│   │   └── training_view.py
│   └── utils/
│       ├── paths.py
│       ├── metadata.py
│       └── seed.py
├── assets/
│   ├── workflows/                ← NUOVO: workflow ComfyUI JSON
│   │   ├── base_txt2img.json
│   │   ├── character_in_pose.json
│   │   ├── product_in_scene.json
│   │   └── correct_inpaint.json
│   ├── styles/dark.qss
│   └── icons/
└── tests/
```

---

## Realismo sulla qualità (onestà necessaria)

Lo stato dell'arte oggi, anche con pipeline ottimale:
- ~75-85% degli output usabili senza ritocchi
- ~15% richiedono piccola correzione (gestita in-app con Ricetta C)
- ~5-10% da rigenerare con altro seed

Questo è meglio di Midjourney/Firefly sul dominio specifico di Bru,
peggio di un grafico umano esperto sul singolo pezzo critico. La forza
del sistema è il VOLUME a qualità costante e la coerenza stilistica,
non la perfezione del singolo output. Le correzioni che servono sono
immediate (un click) invece di richiedere altri software.

Sui difetti tipici (mani, volti): Adetailer li riduce drasticamente ma
non al 100%. È il limite della tecnologia attuale, non
dell'implementazione.

---

## Vincoli e rischi

**VRAM 8GB con pipeline composite.** I moduli si caricano on-demand,
mai tutti insieme. Picco ~7.5GB durante Ricetta A completa. Stretto.
Mitigazioni in docs/ARCHITECTURE.md. Se un workflow specifico va OOM,
ComfyUI gestisce model offloading automatico (più lento ma funziona).

**ComfyUI come dipendenza.** È software di terzi (GPL), embeddato.
Stabile e attivissimo, ma aggiornamenti possono cambiare API. Si pinna
una versione nota-funzionante e si aggiorna con cautela. Vedi
docs/COMFY_ENGINE.md.

**Tempi di setup primo avvio.** Pipeline completa richiede ~15-20GB di
modelli (base + ControlNet + IP-Adapter + ausiliari). Download una
tantum, poi offline. Va comunicato bene.

**Complessità manutenzione.** Sistema più ambizioso = più superfici di
errore. Mitigato da: architettura a moduli isolati, mock mode per dev,
workflow versionati, logging dettagliato.

**Licenze.** Pony (Fair AI), ControlNet (vari, perlopiù permissivi),
IP-Adapter (Apache), ComfyUI (GPL). Per uso personale Bru: tutto ok.
Per distribuzione/commerciale: verifica in docs/MODELS.md.

---

## Primi passi

Ordine di lettura consigliato:
1. HANDOFF.md (questo)
2. docs/ARCHITECTURE.md
3. docs/COMFY_ENGINE.md — il cuore nuovo del sistema
4. docs/RECIPES.md — le tre pipeline nel dettaglio
5. docs/POSE_LIBRARY.md
6. docs/TRAINING.md
7. docs/MODELS.md
8. docs/DEVELOPMENT.md

Poi si parte da Fase 0.
