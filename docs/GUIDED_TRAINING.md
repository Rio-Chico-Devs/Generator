# Training Guidato — Sistema di Apprendimento Partecipativo

> **Stato**: Documento di design. Nessun codice ancora scritto.
> Riferimento per l'implementazione futura di `src/ui/guided_training_view.py`
> e `src/core/guidance/`.

---

## Sommario esecutivo

Il training guidato è una **sessione di dialogo strutturato** tra te e il
modello: il sistema genera candidati uno step alla volta, tu approvi o rifiuti,
il sistema impara **in tempo reale** dalla tua scelta e nel tempo costruisce un
archivio di preferenze (Decision Diary) che alimenterà training futuri più
precisi.

L'idea centrale è che *tu sei l'unico che conosce il risultato giusto*. Nessun
algoritmo può sostituire il tuo occhio. Ma puoi insegnarlo — se hai uno
strumento che registra ogni tua scelta in modo riutilizzabile.

---

## I quattro pilastri

| Pilastro | Cosa fa | Dove vive |
|---|---|---|
| **A — Step Pipeline** | Genera N candidati per ogni fase del disegno, pausa per approvazione | `GuidedSession` + ComfyUI partial-denoise |
| **B — Libreria Tecniche** | Immagini di riferimento per colorazione, ombre, texture: condizionamento via IP-Adapter | `TechniqueLibrary` + profili taggati |
| **C — Decision Diary** | Registra ogni approvazione/rifiuto come coppia (vinto, perso) per DPO futuro | `DiaryEntry` dataclass + JSONL append |
| **D — Due Modalità** | Training: step-by-step, partecipativo. Uso: genera e basta | `mode` flag su project / UI routing |

---

## Pilastro A — Step Pipeline

### Concetto

Un disegno non nasce tutto in una volta. Si costruisce per fasi:
1. **Composizione** — posa, proporzioni, sfondo
2. **Linework** — contorni, dettagli anatomici
3. **Flat colors** — campiture di base
4. **Ombre e luci** — volume, profondità
5. **Dettagli finali** — texture, occhi, capelli, rifinitura

In modalità Uso, ComfyUI fa tutto in un passaggio. In modalità Training, la
pipeline si *ferma* dopo ogni fase e aspetta la tua decisione.

### Meccanismo tecnico

ComfyUI supporta la **denoise parziale** (`denoise_strength` < 1.0 su un
latent esistente). Questo ci permette di:

```
Step 1: genera da noise completo → candidati composizione (denoise 1.0, ~15 step)
        ↓ aspetta approvazione
Step 2: parte dal latent approvato → candidati linework (denoise 0.6, ~10 step)
        ↓ aspetta approvazione
Step 3: parte dal latent approvato → candidati flat colors (denoise 0.5, ~10 step)
        ↓ aspetta approvazione
...e così via
```

L'utente approva **uno** candidato (o rigetta tutti → vedi stato "esaurito").
Il latent dell'approvato diventa il **punto di partenza** per lo step
successivo. Questo è il ciclo fondamentale.

### Struttura dati: GuidedSession

```python
# src/core/guidance/session.py

@dataclass
class StepDefinition:
    id: str                    # "composition", "linework", "flat_colors", ...
    label: str                 # "Composizione", "Linework", ...
    denoise_strength: float    # 1.0 per primo step, 0.4-0.7 per successivi
    n_candidates: int          # quanti candidati generare (default: 4)
    guidance_scale: float      # CFG scale per questo step
    steps: int                 # diffusion steps per questo step
    technique_slots: list[str] # quali slot libreria tecniche usare (vedi Pilastro B)
    # "[]" = nessuna tecnica obbligatoria, ["skin", "shadow"] = usa questi

@dataclass
class Candidate:
    index: int
    image_path: Path
    latent_path: Path          # .pt salvato da ComfyUI custom node
    sidecar_path: Path         # .json con parametri completi
    is_approved: Optional[bool] = None  # None = non ancora valutato

@dataclass
class StepResult:
    step_def: StepDefinition
    candidates: list[Candidate]
    approved_index: Optional[int] = None   # None = tutti rifiutati
    rejection_reason: Optional[str] = None # testo libero opzionale

@dataclass
class GuidedSession:
    session_id: str            # uuid4
    project_slug: str
    created_at: str            # ISO UTC
    prompt: str                # prompt base per tutta la sessione
    negative_prompt: str
    seed: int
    lora_checkpoint: str       # checkpoint LoRA usato
    pipeline: list[StepDefinition]   # sequenza degli step
    results: list[StepResult]  # uno per step completato
    status: str                # "active" | "completed" | "exhausted" | "aborted"
    final_image_path: Optional[Path] = None
```

La sessione viene serializzata come JSON in:
```
projects/{slug}/training/guided/{session_id}/session.json
```

Le immagini candidato vivono in:
```
projects/{slug}/training/guided/{session_id}/step_{N}/candidate_{K}.png
projects/{slug}/training/guided/{session_id}/step_{N}/candidate_{K}.latent.pt
projects/{slug}/training/guided/{session_id}/step_{N}/candidate_{K}.json
```

### Pipeline predefinita (SDXL)

Quattro step standard. L'utente li può personalizzare in un futuro pannello
"Custom Pipeline" (fuori scope prima versione).

```python
DEFAULT_PIPELINE = [
    StepDefinition(
        id="composition",
        label="Composizione",
        denoise_strength=1.0,
        n_candidates=4,
        guidance_scale=7.0,
        steps=20,
        technique_slots=[],
    ),
    StepDefinition(
        id="structure",
        label="Struttura & Linework",
        denoise_strength=0.65,
        n_candidates=4,
        guidance_scale=6.0,
        steps=15,
        technique_slots=["linework"],
    ),
    StepDefinition(
        id="colors",
        label="Colori di base",
        denoise_strength=0.55,
        n_candidates=4,
        guidance_scale=5.5,
        steps=12,
        technique_slots=["skin", "hair"],
    ),
    StepDefinition(
        id="shading",
        label="Ombre e volume",
        denoise_strength=0.45,
        n_candidates=4,
        guidance_scale=5.0,
        steps=10,
        technique_slots=["shadow", "light"],
    ),
]
```

### Flusso UI

```
┌──────────────────────────────────────────────────────────┐
│  TRAINING GUIDATO — Composizione  [step 1/4]             │
│                                                          │
│  Prompt: [iris standing, full body, white background   ] │
│                                                          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐               │
│  │  A   │  │  B   │  │  C   │  │  D   │               │
│  │      │  │      │  │      │  │      │               │
│  │      │  │      │  │      │  │      │               │
│  └──────┘  └──────┘  └──────┘  └──────┘               │
│     ✓          ✓                                        │
│  [Approva A]  [Approva B]  [Approva C]  [Approva D]    │
│                                                          │
│  [Rigenera tutti]  [Cambia tecnica…]  [Annulla sessione] │
│                                                          │
│  Motivo rifiuto (opzionale): [___________________]       │
└──────────────────────────────────────────────────────────┘
```

- Cliccare su una miniatura la ingrandisce in overlay.
- Cliccare "Approva X" → il sistema salva la scelta nel Diary, passa al
  prossimo step usando il latent di X.
- "Rigenera tutti" → conta come N rifiuti (vedi Pilastro C) e riparte.
- "Cambia tecnica…" → apre pannello Libreria Tecniche (Pilastro B) senza
  perdere il progresso.
- Il tasto "Motivo rifiuto" è opzionale ma fortemente suggerito: il testo
  viene salvato nel Diary e può essere usato come segnale negativo futuro.

---

## Pilastro B — Libreria Tecniche

### Concetto

La colorazione è una **tecnica artistica** che richiede coerenza: se la skin
di Iris è sempre ambrata con ombre calde, il modello deve vederla abbastanza
spesso da imparare quella scelta specifica — non solo come pixel casuali ma
come *esempio di riferimento* attivo durante la generazione.

La Libreria Tecniche è una raccolta di immagini che tu classifichi per
categoria. Durante la generazione guidata, le immagini rilevanti vengono
passate a **IP-Adapter** come riferimento visivo.

### Categorie predefinite

| Slot ID | Label | Esempi di contenuto |
|---|---|---|
| `skin` | Colorazione pelle | Swatch di toni, zone con luce/ombra |
| `shadow` | Ombre e profondità | Reference di ombre dure/morbide |
| `light` | Sorgenti luce | Controluce, rim light, ambient |
| `hair` | Capelli & texture | Ciocche, brillantezza, colore |
| `linework` | Linee & contorni | Spessore, qualità linea, dettaglio |
| `background` | Sfondi | Ambienti, texture, profondità |
| `palette` | Palette cromatica | Combinazioni di colori, mood |

Puoi aggiungere slot custom (es. `eyes`, `fabric`, `pose`).

### Struttura dati: TechniqueLibrary

```python
# src/core/guidance/technique_library.py

@dataclass
class TechniqueRef:
    ref_id: str           # uuid4
    slot: str             # "skin", "shadow", ecc.
    image_path: Path
    label: str            # nome human-readable ("Ambra calda + ombre viola")
    notes: str            # note libere ("usare per scene indoor diurne")
    weight: float = 1.0   # peso IP-Adapter (0.0-2.0, default 1.0)
    created_at: str = ""

@dataclass
class TechniqueLibrary:
    project_slug: str
    refs: list[TechniqueRef]

    def by_slot(self, slot: str) -> list[TechniqueRef]: ...
    def active_for_step(self, step_def: StepDefinition) -> list[TechniqueRef]: ...
```

Persistenza in:
```
projects/{slug}/techniques/library.json          # metadati
projects/{slug}/techniques/refs/{ref_id}.png     # immagini originali
projects/{slug}/techniques/refs/{ref_id}.thumb.png
```

### Integrazione con ComfyUI

Per ogni step che ha `technique_slots` non vuoti:
1. Carica i `TechniqueRef` rilevanti dalla libreria
2. Costruisce un workflow ComfyUI che include nodi IP-Adapter:
   - `IPAdapterModelLoader` + `IPAdapterApply` (o `IPAdapterAdvanced`)
   - L'immagine di riferimento viene passata come `image` al nodo
   - Il `weight` del TechniqueRef controlla quanto influenza la generazione
3. Possono esserci più IP-Adapter in parallelo (uno per slot)

Questo significa che quando il sistema genera i candidati per "Colori di base",
le immagini di riferimento `skin` e `hair` della tua libreria vengono *attivamente
usate* per condizionare l'output — non solo come esempi passati a training, ma
come guida in tempo reale.

### UI: Pannello Libreria Tecniche

```
┌─────────────────────────────────────────┐
│  LIBRERIA TECNICHE                      │
│                                         │
│  Slot: [skin ▾]                         │
│                                         │
│  ┌────┐ ┌────┐ ┌────┐  [+ Aggiungi]    │
│  │ref1│ │ref2│ │ref3│                   │
│  │    │ │    │ │    │                   │
│  └────┘ └────┘ └────┘                   │
│  "Ambra" "Rosa" "Oliva"                 │
│                                         │
│  Peso IP-Adapter: [===|======] 1.0      │
│                                         │
│  [Modifica]  [Rimuovi]  [Chiudi]        │
└─────────────────────────────────────────┘
```

Accessibile dalla barra laterale anche fuori dal training guidato, così puoi
costruire la libreria in modo incrementale mentre generi immagini normalmente.

Quando trovi un'immagine generata con una colorazione che ti piace → tasto
"→ Libreria tecniche" (accanto a "→ Dataset") nella Gallery.

---

## Pilastro C — Decision Diary

### Concetto

Ogni tua scelta — "A è meglio di B" — è un **dato prezioso**. Non va sprecato.
Il Decision Diary lo cattura in formato strutturato compatibile con DPO
(Direct Preference Optimization), lo standard attuale per training da
preferenze umane.

Oggi questi dati alimentano solo la scelta del checkpoint e le statistiche. In
futuro potranno essere usati per fine-tuning DPO del LoRA stesso — ma anche
senza DPO, sono già utili per capire "cosa funziona" e prendere decisioni
consapevoli.

### Struttura dati: DiaryEntry

```python
# src/core/guidance/diary.py

@dataclass
class DiaryEntry:
    entry_id: str          # uuid4
    session_id: str        # riferimento alla GuidedSession
    project_slug: str
    timestamp: str         # ISO UTC
    step_id: str           # "composition", "colors", ecc.

    # DPO pair: chosen > rejected
    chosen_path: Optional[Path]    # immagine approvata (None se tutti rifiutati)
    rejected_paths: list[Path]     # immagini rifiutate

    # Prompt usato in questo step
    prompt: str
    negative_prompt: str
    seed: int

    # Metadati aggiuntivi
    rejection_reason: Optional[str]  # testo libero utente
    technique_refs_used: list[str]   # ref_id delle tecniche attive
    lora_checkpoint: str

    # Segnali derivati
    was_regeneration: bool  # True se l'utente ha premuto "Rigenera tutti"
```

Ogni entry viene appesa a un file JSONL (una entry per riga):
```
projects/{slug}/training/guided/diary.jsonl
```

JSONL è la scelta giusta: append-only, mai riscrivere, leggibile riga per
riga senza caricare tutto in memoria, compatibile con i tool DPO della
community (TRL, axolotl).

### Statistiche derivabili dal Diary

Senza DPO, il Diary permette già analisi potenti:

```python
def diary_stats(entries: list[DiaryEntry]) -> dict:
    return {
        "total_sessions": ...,
        "approval_rate": ...,           # % di step con almeno un approvato
        "avg_rejections_before_approval": ...,
        "most_rejected_step": ...,      # quale step causa più problemi
        "best_technique_combos": ...,   # quali tecniche hanno più approvazioni
        "rejection_reason_tags": ...,   # Counter dei motivi frequenti
    }
```

Questo diventa il pannello **"Cosa ho imparato"** nella sezione Training:
un resoconto leggibile di cosa funziona e cosa no nel tuo stile.

### Stato "Esaurito" — quando il sistema si arrende

**Definizione:** se in uno stesso step l'utente rifiuta tutti i candidati per
K volte consecutive (K = 3 di default, configurabile), il sistema entra in
stato `exhausted`.

**Cosa succede:**

```
┌────────────────────────────────────────────────┐
│  ⚠  Soluzioni Insufficienti                    │
│                                                │
│  Ho generato 12 candidati per "Colori di base" │
│  ma nessuno ti ha convinto.                    │
│                                                │
│  Questo può significare:                       │
│  → Il dataset manca di esempi di questa tecnica│
│  → Il LoRA attuale non ha visto abbastanza     │
│    riferimenti di colorazione calda            │
│  → La tecnica richiesta va aggiunta alla       │
│    Libreria Tecniche prima di riprovare        │
│                                                │
│  Cosa vuoi fare?                               │
│  [Aggiungi esempi alla Libreria Tecniche]      │
│  [Aggiungi immagini al Dataset e re-train]     │
│  [Accetta il migliore tra questi ▾]            │
│  [Annulla questa sessione]                     │
└────────────────────────────────────────────────┘
```

L'entry nel Diary viene marcata `status = "exhausted"` con `step_id` e
tutti i candidati generati come `rejected_paths`. Questa è informazione
preziosa: sa esattamente dove il LoRA fallisce.

**Nota importante:** "Esaurito" non è un fallimento del sistema, è una
diagnosi. Il sistema ti sta dicendo "non ho abbastanza per fare quello che
chiedi — ecco di cosa ho bisogno". È il comportamento corretto.

---

## Pilastro D — Due modalità

### Modalità Uso (default)

- Nessuna pausa, nessuna approvazione
- Genera con il LoRA attivo, seed casuale o bloccato
- Vedi la gallery, tagga con 👍/🔀/👎
- Il Diary **non registra** nulla (nessun "expected choice")
- Accessibile dalla tab "Genera" come oggi

### Modalità Training Guidato

- Ogni sessione è un contratto: tu guidi, il sistema registra
- Accessibile dalla nuova tab "Training Guidato" (o sotto-tab di Training)
- Richiede:
  - LoRA attivo nel progetto (o warning "addestra prima un LoRA base")
  - Almeno una voce per slot nella Libreria Tecniche (o warning "aggiungi
    riferimenti" con opzione di procedere comunque senza IP-Adapter)
- Lo stato della sessione è persistito: se chiudi l'app a metà step, puoi
  riprendere

### Routing nella UI

```python
# src/ui/main_window.py — tab routing

if mode == "guided_training":
    show(GuidedTrainingView)
else:
    show(GenerateView)  # modalità uso attuale
```

Il `Project` acquisisce un flag:
```python
@dataclass
class Project:
    ...
    training_mode: str = "use"  # "use" | "guided"
```

Il cambio di modo è visibile in `ProjectSettingsDialog` (non in main nav per
evitare switch accidentale).

---

## Integrazione tra i pilastri

```
                    ┌──────────────────────┐
                    │   GuidedTrainingView  │
                    │    (Pilastro D UI)    │
                    └──────────┬───────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
    ┌──────────▼───────────┐       ┌───────────▼──────────┐
    │    GuidedSession      │       │   TechniqueLibrary    │
    │    (Pilastro A)       │◄──────│    (Pilastro B)       │
    │                      │       │                       │
    │  - step pipeline     │       │  - slot refs          │
    │  - latent handoff    │       │  - IP-Adapter weights │
    │  - candidate grid    │       │  - by_slot()          │
    └──────────┬───────────┘       └───────────────────────┘
               │ ogni approvazione/rifiuto
               ▼
    ┌──────────────────────┐
    │    DecisionDiary      │
    │    (Pilastro C)       │
    │                      │
    │  - DiaryEntry append  │
    │  - exhausted check   │
    │  - stats()           │
    └──────────────────────┘
               │ alimenta
               ▼
    ┌──────────────────────┐       ┌──────────────────────┐
    │  DatasetInspector    │       │  DPO Training        │
    │  "Cosa ha capito"    │       │  (futuro)            │
    │  + analisi lacune    │       │  diary.jsonl → DPO   │
    └──────────────────────┘       └──────────────────────┘
```

### Flusso di una sessione completa

```
1. Utente apre "Training Guidato"
   → scegli prompt, LoRA, seed
   → sistema carica DEFAULT_PIPELINE
   → sistema carica TechniqueLibrary dal progetto

2. Step 1 — Composizione
   → ComfyUI genera 4 candidati (denoise 1.0)
   → UI mostra grid candidati
   → [se tecnica slots vuoti → nessun IP-Adapter]
   → Utente approva "C"
   → DiaryEntry: chosen=C, rejected=[A,B,D]
   → sistema salva latent di C

3. Step 2 — Struttura
   → ComfyUI parte dal latent di C, denoise 0.65
   → se library ha refs per slot "linework" → IP-Adapter attivo
   → Utente rifiuta tutti → "Rigenera tutti"
   → DiaryEntry: chosen=None, rejected=[...], was_regeneration=True
   → seconda round → utente approva "B"
   → DiaryEntry: chosen=B, rejected=[A,C,D]

4. [... step 3, step 4 ...]

5. Step finale → immagine completata
   → sessione.status = "completed"
   → immagine copiata in gallery con tag automatico 🏅 "guided"
   → Diary mostra riepilogo sessione
```

---

## Build order — ordine di costruzione

Le dipendenze sono chiare. Costruire dal basso verso l'alto:

### Fase 1 — Fondamenta dati (nessuna UI, solo `src/core/guidance/`)

**1.1** `diary.py` — `DiaryEntry` dataclass + `append_entry()` + `load_diary()`
+ `diary_stats()`. Nessuna dipendenza esterna. Testabile subito.

**1.2** `session.py` — `StepDefinition`, `Candidate`, `StepResult`,
`GuidedSession`. Serializzazione/deserializzazione JSON. `DEFAULT_PIPELINE`.
Dipende da: niente (solo pathlib).

**1.3** `technique_library.py` — `TechniqueRef`, `TechniqueLibrary`,
`by_slot()`, `active_for_step()`. Persistenza library.json.
Dipende da: session.StepDefinition.

**1.4** `exhaustion.py` — logica di rilevamento stato esaurito.
`check_exhaustion(session, step_id, k=3) -> bool`. Pura logica, nessuna UI.
Dipende da: session.GuidedSession.

**Test Fase 1:** ~30 test unitari su dati fittizi, nessun ComfyUI.

---

### Fase 2 — Integrazione ComfyUI

**2.1** Workflow ComfyUI per "denoise parziale da latent esistente".
Il latent viene salvato/caricato come `.pt` tramite nodo custom o via API
ComfyUI `SaveLatent` / `LoadLatent`. Questo richiede verifica che il
nodo esista nella versione embedded — o implementarlo.

**2.2** Wrapper `GuidedWorker(QThread)` — esegue uno step della pipeline:
- riceve `GuidedSession` + `step_index` + `TechniqueLibrary`
- costruisce workflow ComfyUI (con IP-Adapter se lib non vuota)
- monitora output, emette `candidate_ready(index, path)` per ogni candidato
- emette `step_complete()` quando tutti i candidati sono pronti

**2.3** Modifica `ComfyEngine` per supportare:
- salvataggio latent intermedi
- caricamento latent per denoise parziale
- IP-Adapter multi-slot

---

### Fase 3 — UI step-by-step

**3.1** `GuidedTrainingView` — widget principale.
- Intestazione: prompt, step indicator "Step 2/4 — Struttura"
- Area candidati: griglia 2×2 di `CandidateThumb`
- Pannello azioni: Approva, Rigenera, Cambia tecnica, Motivo rifiuto
- Barra progresso sessione

**3.2** `CandidateThumb` — miniatura con overlay approvazione.

**3.3** Integrazione `check_exhaustion()`: quando K rifiuti consecutivi →
mostra `ExhaustedDialog` con le 4 opzioni.

---

### Fase 4 — Libreria Tecniche UI

**4.1** `TechniqueLibraryView` — pannello standalone slot+refs+peso.
Accessibile da sidebar progetto.

**4.2** Pulsante "→ Libreria tecniche" in `GalleryView` detail panel
(accanto ai bottoni rating esistenti).

**4.3** `TechniquePickerDialog` — selettore inline aperto da "Cambia tecnica…"
durante la sessione guidata.

---

### Fase 5 — Dashboard "Cosa ho imparato"

**5.1** `DiaryStatsView` — tab in Training.
- Approval rate per step (bar chart)
- Top motivi di rifiuto (word cloud / tag table)
- Tecniche più usate / più efficaci

**5.2** Integrazione con `DatasetInspector` già esistente: banner "hai N sessioni
con rifiuti su 'skin' — considera di aggiungere più esempi a questa categoria".

---

### Fase 6 — DPO Training (futuro, fuori scope v1)

`diary.jsonl` → script di conversione in formato `trl` DPO dataset →
fine-tuning LoRA con preferenze. Documentato ma non implementato nella v1.

---

## Dipendenze tecniche da verificare

### ComfyUI custom nodes necessari

| Nodo | Funzione | Status |
|---|---|---|
| `SaveLatent` | Salva latent su disco in formato .pt | Da verificare/implementare |
| `LoadLatent` | Carica latent da disco | Da verificare/implementare |
| `IPAdapterModelLoader` | Carica modello IP-Adapter | Community node (ComfyUI_IPAdapter_plus) |
| `IPAdapterApply` / `IPAdapterAdvanced` | Applica condizionamento | Community node |

### Modelli aggiuntivi necessari

| Modello | Dimensione | Funzione |
|---|---|---|
| IP-Adapter SDXL | ~2.5 GB | Condizionamento visivo per SDXL |
| IP-Adapter Plus SDXL | ~3.5 GB | Versione con più fedeltà |
| CLIP Vision ViT-H | ~2.5 GB | Encoder visivo richiesto da IP-Adapter |

Nota: questi si scaricano una sola volta e sono condivisi tra tutti i progetti.

---

## Parametri configurabili (project.json)

Aggiunta futura al `Project`:

```python
@dataclass
class GuidedTrainingConfig:
    exhaustion_threshold: int = 3    # K rifiuti consecutivi → esaurito
    candidates_per_step: int = 4     # N candidati per step
    pipeline: str = "default"        # "default" | custom serialized
    ip_adapter_enabled: bool = True  # usa IP-Adapter se libreria disponibile
    ip_adapter_weight: float = 1.0   # peso globale (sovrascritto da slot)
    save_all_candidates: bool = True # conserva tutte le immagini rifiutate
```

---

## Cosa NON fa il sistema (scope out)

- **Non modifica i pesi del LoRA in tempo reale** durante la sessione guidata.
  La sessione raccoglie preferenze — il re-training avviene dopo, separatamente.
- **Non fa DPO automatico** dopo ogni sessione (v1). Il Diary si accumula
  finché non si sceglie consapevolmente di fare un training DPO.
- **Non ha "undo step"** in v1: una volta approvato un candidato e passato allo
  step successivo, non si torna indietro. (Salvare il latent rende tecnicamente
  possibile implementarlo in futuro.)
- **Non supporta pipeline personalizzata via UI** in v1: si usa DEFAULT_PIPELINE
  o si edita a mano il JSON della sessione.
- **Non esegue pre-screening automatico** (es. face detector, aesthetic scorer)
  prima di mostrare i candidati. Sei tu il filtro.

---

## Note implementative importanti

### Latent persistence è il punto critico

Il passaggio di stato tra step (il latent approvato diventa input del prossimo)
è la feature più delicata tecnicamente. ComfyUI in modalità API non espone
direttamente il latent come array numpy — lo gestisce internamente. Le opzioni:

1. **SaveLatent custom node**: il workflow include un nodo che salva il latent
   su disco; il workflow successivo lo carica con LoadLatent. Richiede nodi
   custom installati nell'engine embedded.

2. **Full image re-encode**: invece del latent, si salva l'immagine PNG e il
   prossimo step la ri-encode in latent con `VAEEncode`. Introduce leggera
   perdita VAE ma è più semplice e non richiede nodi custom.

3. **img2img con alta denoise**: approccio simile al 2 ma usando img2img
   nativo di ComfyUI sul PNG approvato. Standard, robusto.

**Raccomandazione v1**: opzione 3 (img2img su PNG) — più semplice, nessuna
dipendenza extra, qualità accettabile. Passare al latent puro in v2 se
la perdita qualitativa è misurabile.

### IP-Adapter è opzionale, non bloccante

Se l'utente non ha la Libreria Tecniche popolata, il sistema funziona
identicamente — semplicemente senza il condizionamento visivo. Il warning
è informativo, non bloccante. Questo permette di iniziare a usare il Training
Guidato subito, aggiungendo le tecniche man mano.

### Il Diary è append-only

Mai modificare/cancellare entry dal Diary. Se l'utente vuole "dimenticare"
una sessione, si marca come `status = "ignored"` — ma i dati rimangono.
I dati storici hanno sempre valore per analisi retrospettiva.
