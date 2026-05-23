# Training

Documento di riferimento per il sistema di training. Spiega cosa fa
l'app sotto il cofano e come ottimizzare per RTX 3060 / 8GB VRAM.

## Cosa addestriamo (e cosa NO)

**Addestriamo:** un LoRA (Low-Rank Adaptation) sopra un modello base
SDXL o SD 1.5. Il LoRA è un piccolo set di matrici (~100-300 MB) che
modifica il comportamento del modello senza alterarlo.

**Non addestriamo:** il modello base stesso (sarebbe DreamBooth full
fine-tuning, fuori scope per 8GB). Non addestriamo da zero (ci
vorrebbero settimane e migliaia di GPU/h).

Concetto fondamentale: il modello base "sa già disegnare". Il LoRA gli
dice "quando vedi il tag attivatore `vf_iris_v1`, disegna in questo
modo specifico". È molto più efficiente che insegnare da zero, e
funziona splendidamente con 50-200 immagini.

## Engine: sd-scripts (kohya)

L'app usa `sd-scripts` di kohya-ss come engine di training. Motivi:
- Standard de facto della community open source
- Supporta SDXL, SD 1.5, FLUX
- Ottimizzazioni VRAM eccellenti
- Logging dettagliato parsabile
- Battle-tested su milioni di training

Il modulo è incluso nel virtualenv dell'app (non installato globalmente
sul sistema utente). L'app lo chiama via subprocess passando un file
di config TOML generato dinamicamente.

**Vantaggio del subprocess**: se il training crasha o satura VRAM, la
UI sopravvive. Si può killare pulito senza riavviare l'app.

## Pipeline di training (cosa succede quando l'utente clicca "Addestra")

```
[1] PREPARE DATASET                                        (~1 min)
    ├─ Leggi manifest progetto
    ├─ Filtra immagini con excluded_from_training=true
    ├─ Bucket assignment: raggruppa per aspect ratio
    │  (SDXL: bucket 512², 768², 1024², 1024×1536, 1536×1024)
    ├─ Resize/crop a bucket size, salva in processed/
    └─ Verifica tag file esistente per ogni immagine

[2] GENERATE TRAIN CONFIG                                   (<1s)
    Scrivi file TOML in training/runs/{timestamp}/config.toml
    con tutti i parametri del preset selezionato

[3] LAUNCH SUBPROCESS                                       (<5s)
    accelerate launch --num_cpu_threads_per_process 2 \
      sd-scripts/sdxl_train_network.py \
      --config_file training/runs/{ts}/config.toml

[4] MONITOR LOOP                                            (1-4 ore)
    ├─ Parse stdout per progress (current_step / max_steps)
    ├─ Parse loss values, emetti signal per UI graph
    ├─ Quando sd-scripts salva sample images, copia in samples/
    ├─ Quando salva checkpoint, copia in checkpoints/
    └─ Su SIGTERM/abort: cleanup + status.json = "aborted"

[5] FINALIZE                                                (<10s)
    ├─ Verifica checkpoint finale esista
    ├─ Calcola hash safetensors
    ├─ Aggiorna status.json = "done"
    ├─ Aggiorna project.json.active_lora
    └─ Emit signal "training_complete" alla UI
```

## Preset di training

Tre preset bilanciati per 8GB VRAM, scelti per coprire i casi d'uso
tipici senza esporre 30 parametri all'utente.

### Preset "Veloce" (~30-45 min, SD 1.5)

Per esperimenti, prototipi, dataset piccoli (<30 immagini), o per
chi vuole vedere risultati subito prima di committarsi.

```toml
pretrained_model_name_or_path = "{base_model_sd15_path}"
train_data_dir = "{dataset_processed_path}"
output_dir = "{run_dir}/checkpoints"
output_name = "lora"
save_model_as = "safetensors"

resolution = "512,512"
train_batch_size = 1
gradient_accumulation_steps = 1
max_train_epochs = 10
save_every_n_epochs = 2

network_module = "networks.lora"
network_dim = 16
network_alpha = 8
network_train_unet_only = true   # NO text encoder = -1.5GB VRAM

learning_rate = 1e-4
unet_lr = 1e-4
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 50

mixed_precision = "fp16"
optimizer_type = "AdamW8bit"
xformers = true
gradient_checkpointing = true
cache_latents = true

sample_every_n_epochs = 2
sample_prompts = "{prompts_file}"
sample_sampler = "dpm_2"

seed = 42
clip_skip = 2
```

VRAM picco: ~5GB. Tempo: 30-45 min su 100 immagini.

### Preset "Standard" (~1.5-2.5 ore, SDXL)

Default. Bilanciato per dataset 50-150 immagini, qualità di output
elevata, tempo accettabile.

```toml
pretrained_model_name_or_path = "{base_model_sdxl_path}"
train_data_dir = "{dataset_processed_path}"
output_dir = "{run_dir}/checkpoints"
output_name = "lora"
save_model_as = "safetensors"

resolution = "1024,1024"
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64

train_batch_size = 1
gradient_accumulation_steps = 4
max_train_epochs = 10
save_every_n_epochs = 2

network_module = "networks.lora"
network_dim = 16
network_alpha = 8
network_train_unet_only = true

learning_rate = 1e-4
unet_lr = 1e-4
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

mixed_precision = "bf16"             # bf16 > fp16 su SDXL per stabilità
optimizer_type = "AdamW8bit"
xformers = true
gradient_checkpointing = true
cache_latents = true
cache_latents_to_disk = true         # cruciale per VRAM
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true

no_half_vae = true                   # SDXL VAE non gestisce fp16 bene
sample_every_n_epochs = 2
sample_prompts = "{prompts_file}"
sample_sampler = "dpmpp_2m"

seed = 42
max_data_loader_n_workers = 0        # Windows: 0 obbligatorio
```

VRAM picco: ~7.5GB. Tempo: ~2 ore per 100 immagini.

### Preset "Massimo" (~3-5 ore, SDXL high rank)

Per dataset grandi (150+) o quando si vuole massima fedeltà stilistica.
Rank LoRA più alto → cattura più sfumature → file più grande (~300MB).

Differenze da Standard:
```toml
network_dim = 32          # rank raddoppiato
network_alpha = 16
max_train_epochs = 15
gradient_accumulation_steps = 8   # batch effettivo 8, più stabile
learning_rate = 5e-5      # lr più basso, training più lungo ma più preciso
```

VRAM picco: ~7.8GB (al limite). Tempo: 3-5 ore.

## Parametri esposti in modalità Advanced

L'utente normale vede solo: preset + numero epoch + "Inizia". Power user
può aprire pannello Advanced e modificare:
- `network_dim` (4-128)
- `network_alpha` (suggerito = dim/2)
- `learning_rate` (1e-5 - 5e-4)
- `train_batch_size` × `gradient_accumulation_steps`
- `lr_scheduler` (cosine, constant, linear, cosine_with_restarts)
- `optimizer_type` (AdamW8bit, Prodigy, DAdaptation)
- `train_text_encoder` (con warning VRAM)
- `noise_offset` (0-0.1, per output con più contrasto)
- `min_snr_gamma` (5 default, migliora convergenza)

## Sample images durante training

A intervalli configurabili sd-scripts genera immagini sample con il LoRA
"in fase di cottura" usando prompt definiti dall'utente.

Sono fondamentali perché permettono di:
1. Vedere il progresso in tempo reale
2. Capire quando lo stile "si è formato" (epoch 4? 6? 10?)
3. Scegliere il checkpoint giusto (a volte step 6000 è meglio del finale)
4. Abortire presto se qualcosa va male (overfitting visibile)

L'app genera automaticamente sample prompts intelligenti dal dataset:
```
{activator_tag}, 1girl, simple pose
{activator_tag}, full body, dynamic angle
{activator_tag}, portrait, soft lighting
```
L'utente può sostituirli con prompt custom dal dataset view.

## Indicatori di problemi

| Sintomo | Probabile causa | Soluzione |
|---|---|---|
| Loss oscilla violentemente | learning rate troppo alto | Riduci a 5e-5 |
| Loss scende rapidamente poi piatta | overfitting precoce | Più dataset / rank più basso |
| Sample images identiche tra epoch | learning rate troppo basso | Aumenta a 2e-4 |
| Sample images deformi/cursed | rank troppo alto per dataset piccolo | network_dim 8, alpha 4 |
| OOM dopo N step | VAE tiling disabilitato accidentalmente | Verifica `cache_latents = true` |
| Output ignora il tag attivatore | tag attivatore non in cima ai .txt | Re-tag automatico |
| Output troppo "anime" anche per altre cose | base model Pony/Animagine, manca diversificazione dataset | Cambia base a Juggernaut, o aggiungi immagini più varie |

L'app monitorerà loss in real-time e farà warning automatici in caso di
pattern problematici noti.

## Checkpoint selection

Quando un training finisce con `save_every_n_epochs = 2`, hai N
checkpoint intermedi + uno finale. NON sempre il finale è il migliore:
- Troppo training → "overfitting": il LoRA replica le immagini del
  dataset invece di generalizzare lo stile
- Troppo poco training → "underfitting": lo stile non è ancora forte

L'app mostra in **Run Detail View** una griglia con sample image di
ogni checkpoint. L'utente sceglie quale promuovere come `active_lora`.
Si può cambiare in ogni momento senza re-training.

## Resume da crash

Se un training crasha (OOM, blackout, crash app), `status.json`
contiene `current_step` e `last_checkpoint_path`. L'utente, alla
ri-apertura del progetto, vede un banner: "Training interrotto allo
step X. Riprendi?". Il resume passa `--resume {last_checkpoint_path}` a
sd-scripts.

## Sicurezza training

- Subprocess girato con working directory isolato (`run_dir`)
- Limit massimo `max_train_steps` calcolato dall'utente, non da config
  esterno (no risk di training infiniti)
- Limit massimo network_dim a 128 (evita LoRA enormi accidentali)
- Pre-flight check VRAM disponibile prima di lanciare (almeno 7GB
  liberi su 8GB totali)
- File config TOML generato sempre da template + parametri validati,
  mai assemblato da string concatenation
