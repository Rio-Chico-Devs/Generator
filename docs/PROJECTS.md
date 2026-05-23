# Anatomia di un progetto

Un Progetto è l'unità fondamentale di Vihente Forge. Tutto ruota attorno
a questo concetto. Capirlo bene ora evita refactor dopo.

## Definizione

Un Progetto è una **cartella autonoma** sul filesystem che contiene tutto
ciò che riguarda uno stile artistico specifico: dataset sorgente, file di
configurazione, runs di training, LoRA addestrati, immagini generate,
prompt history.

Niente database centrale. Niente registry. Niente lock file globali. Un
progetto è una cartella, e basta. Si può zip-are e spostare, si può
backup-are con `cp -r`, si può aprire un giorno qualsiasi in futuro
anche senza l'app (i file sono leggibili).

## Path di default

`~/Documents/Vihente Forge/projects/{slug_progetto}/`

`slug_progetto` è derivato dal nome user (es. "Iris Vihente" → `iris-vihente`).
Conflitti su filesystem risolti con suffisso numerico.

L'utente può cambiare la root in Settings; vale per i nuovi progetti, i
vecchi restano dove sono.

## Struttura completa

```
iris-vihente/
├── project.json                      # ← Metadata principale (vedi sotto)
├── README.md                         # ← Auto-generato, descrive il progetto
│
├── dataset/
│   ├── images/                       # File originali, mai modificati
│   │   ├── img_0001.png
│   │   ├── img_0002.jpg
│   │   └── img_0003.webp
│   ├── processed/                    # Versioni preparate per training
│   │   ├── img_0001.png              # Resize/crop secondo bucket
│   │   └── img_0001.txt              # Tag file
│   ├── thumbnails/                   # 256px per UI griglia
│   │   └── img_0001.jpg
│   └── manifest.json                 # Stato di ogni immagine (vedi sotto)
│
├── training/
│   ├── runs/
│   │   ├── 2026-05-19_001_standard/
│   │   │   ├── config.json           # Parametri esatti usati
│   │   │   ├── samples/              # Sample generati durante training
│   │   │   │   ├── step_000500.png
│   │   │   │   ├── step_001000.png
│   │   │   │   └── ...
│   │   │   ├── checkpoints/
│   │   │   │   ├── lora_step_001000.safetensors
│   │   │   │   ├── lora_step_002000.safetensors
│   │   │   │   └── lora_final.safetensors
│   │   │   ├── log.txt               # Log completo training
│   │   │   └── status.json           # running / done / failed / aborted
│   │   └── 2026-05-22_002_massimo/
│   │       └── ...
│   └── active_lora.json              # Punta al LoRA in uso
│
├── generations/                      # Tutte le generazioni del progetto
│   └── 2026-05-19/
│       ├── 142301_a7b3c1.png
│       ├── 142301_a7b3c1.json
│       ├── 142545_f8e2d4.png
│       └── 142545_f8e2d4.json
│
├── gallery/                          # Immagini "promosse" dall'utente
│   ├── 142301_a7b3c1.png             # Symlink o copia da generations/
│   └── ...
│
└── prompts/
    ├── favorites.json                # Prompt salvati dall'utente
    └── history.json                  # Tutti i prompt usati con timestamp
```

## `project.json` — struttura

```json
{
  "schema_version": 1,
  "name": "Iris Vihente",
  "slug": "iris-vihente",
  "description": "Character art per il progetto Vihente / Iris",
  "created_at": "2026-05-19T14:23:01Z",
  "updated_at": "2026-05-19T18:45:33Z",

  "base_model": {
    "id": "pony-v6-xl",
    "name": "Pony Diffusion V6 XL",
    "hash": "67ab2fd8...",
    "locked": true
  },

  "activator_tag": "vf_iris_v1",

  "trigger_prompt_prefix": "vf_iris_v1, masterpiece, best quality,",
  "default_negative_prompt": "low quality, worst quality, blurry, deformed",

  "default_generation_params": {
    "steps": 30,
    "cfg_scale": 7.0,
    "sampler": "dpmpp_2m_karras",
    "width": 1024,
    "height": 1024
  },

  "active_lora": {
    "run_id": "2026-05-19_001_standard",
    "checkpoint": "lora_final.safetensors",
    "weight": 0.85
  },

  "stats": {
    "dataset_image_count": 127,
    "training_runs": 1,
    "generation_count": 43,
    "gallery_count": 8
  }
}
```

Campi notabili:
- **`schema_version`** — per migrazioni future quando cambieremo lo schema
- **`base_model.locked`** — true blocca cambio del modello base (default).
  L'utente deve sbloccarlo esplicitamente in advanced (con warning di
  invalidazione LoRA)
- **`activator_tag`** — generato automaticamente, immutabile dopo
  training (cambiarlo richiede re-tag dataset)
- **`active_lora`** — punta al run + checkpoint specifico; l'utente
  può tornare a un checkpoint precedente se uno step intermedio era
  migliore del finale

## `dataset/manifest.json` — struttura

```json
{
  "schema_version": 1,
  "images": {
    "img_0001": {
      "original_filename": "iris_full_body_001.png",
      "original_path": "images/img_0001.png",
      "processed_path": "processed/img_0001.png",
      "thumbnail_path": "thumbnails/img_0001.jpg",
      "tags_path": "processed/img_0001.txt",
      "added_at": "2026-05-19T14:30:00Z",
      "original_size": [1920, 1280],
      "processed_size": [1024, 1024],
      "bucket": "1024x1024",
      "tags": ["vf_iris_v1", "1girl", "long hair", "..."],
      "tags_edited_by_user": true,
      "excluded_from_training": false,
      "quality_score": null,
      "checksum_sha256": "a7b3c1d4..."
    }
  }
}
```

Campi notabili:
- **`bucket`** — sd-scripts usa "aspect ratio bucketing": immagini di
  proporzioni simili vengono raggruppate. Il manifest tiene traccia
  del bucket assegnato per training riproducibili
- **`excluded_from_training`** — l'utente può escludere immagini singole
  senza rimuoverle dal dataset (utile per A/B test)
- **`tags_edited_by_user`** — distingue tag auto-generati da tag corretti
  dall'utente. In re-tagging si preservano le correzioni
- **`checksum_sha256`** — rileva modifiche accidentali ai file originali

## Ciclo di vita di un progetto

```
1. CREATE
   project.json scritto, struttura cartelle creata
   manifest vuoto
   ↓
2. ADD IMAGES
   drag&drop → copiate in dataset/images/
   manifest aggiornato con entry pending
   thumbnail generate in background
   ↓
3. PROCESS DATASET
   resize/crop secondo bucket SDXL
   versioni in dataset/processed/
   ↓
4. AUTO-TAG
   WD-tagger su ogni immagine processed/
   file .txt generati con tag attivatore in cima
   ↓
5. (opzionale) USER REFINE TAGS
   editor UI per correggere tag
   manifest.tags_edited_by_user = true
   ↓
6. TRAIN
   nuovo training/runs/{timestamp}/
   subprocess sd-scripts
   sample + checkpoints salvati progressivamente
   ↓
7. SELECT CHECKPOINT
   l'utente vede sample, sceglie il checkpoint da promuovere
   project.json.active_lora aggiornato
   ↓
8. GENERATE
   ogni immagine generata salvata in generations/{date}/
   con PNG metadata + JSON sidecar
   ↓
9. PROMOTE TO GALLERY
   "questa è davvero buona" → copia/symlink in gallery/
   ↓
10. (opzionale) FEEDBACK LOOP
    immagini in gallery/ possono essere aggiunte al dataset
    nuovo training run con dataset arricchito
```

## Operazioni supportate sui progetti

| Operazione | Comportamento |
|---|---|
| **Create** | Wizard: nome, descrizione, modello base, dimensione preset |
| **Open** | Caricamento `project.json`, validazione schema |
| **Duplicate** | Copia cartella, nuovo slug, nuovo `activator_tag` |
| **Archive** | Sposta in `~/Documents/Vihente Forge/archived/`. Non distrugge nulla |
| **Export** | Crea `{slug}.vforge` (zip con tutto). Opzioni: include/escludi training runs (per ridurre dimensione) |
| **Import** | Apre `.vforge`, validazione schema, ripristino |
| **Delete** | Sposta in trash di sistema. Mai unlink diretto |
| **Rename** | Solo `name`, slug resta immutato (per non rompere path) |

## Validazione progetto

All'apertura, l'app esegue check:
- `project.json` presente e schema valido?
- `base_model` ancora disponibile sul sistema?
- `active_lora` referenziato esiste?
- dataset images conteggiati matchano manifest?
- checksum tag file presenti?

Su check falliti, dialog di repair con opzioni:
- "Continua con limitazioni" (es. modello base mancante → disabilita generation, lascia dataset accessibile)
- "Cerca modello altrove"
- "Ripara automaticamente" (rigenera manifest da filesystem se possibile)

## Migrations

`schema_version` in `project.json` permette migrazioni automatiche:
- App lancia → legge schema_version
- Se < corrente, esegue migrations in sequenza
- Backup automatico in `.vforge_backup_{old_version}.json` prima della migration
- Mai distruttive: rollback sempre possibile

## Convenzioni di naming

- **Cartelle progetto**: `kebab-case` (slug)
- **File immagini originali**: rinominati a `img_NNNN.{ext}` durante import (l'originale è preservato come metadata, ma il sistema usa nomi predicibili)
- **Tag files**: stesso basename dell'immagine corrispondente, estensione `.txt`
- **Run training**: `YYYY-MM-DD_NNN_{preset_name}` (date + counter + preset)
- **Generazioni**: `HHMMSS_{hash6}.png` (orario + hash breve del seed+prompt per univocità)
- **LoRA checkpoints**: `lora_step_NNNNNN.safetensors` o `lora_final.safetensors`
