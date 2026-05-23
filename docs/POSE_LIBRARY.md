# Pose Library

Gestione della collezione di ~500 pose di riferimento di Bru. Sono il
"catalogo di pose pronte" da applicare ai personaggi via ControlNet.

## Concetto chiave

Le pose NON sono un dataset di training. Sono input ControlNet. Lo stile
delle immagini di riferimento è irrilevante: una foto reale, uno sketch
a matita, un manichino 3D — conta solo lo **scheletro** (posizione di
testa, busto, braccia, gambe) che se ne estrae.

Questo significa che Bru può usare QUALSIASI immagine come reference di
posa: foto trovate, propri scatti, disegni, screenshot. Tutte e 500.

## Struttura su disco

```
~/Documents/Vihente Forge/pose_library/
├── library.json              ← indice con metadata di ogni posa
├── sources/                  ← immagini originali (qualsiasi formato/stile)
│   ├── pose_0001.jpg
│   ├── pose_0002.png
│   └── ...
├── skeletons/                ← scheletri pre-estratti (cache)
│   ├── pose_0001.png         ← visualizzazione OpenPose (per preview)
│   └── pose_0001.json        ← dati joint (per ControlNet diretto)
└── thumbnails/               ← preview 256px per la griglia UI
    └── pose_0001.jpg
```

## library.json

```json
{
  "schema_version": 1,
  "poses": {
    "pose_0001": {
      "source_path": "sources/pose_0001.jpg",
      "skeleton_path": "skeletons/pose_0001.json",
      "skeleton_preview": "skeletons/pose_0001.png",
      "thumbnail": "thumbnails/pose_0001.jpg",
      "added_at": "2026-05-22T10:00:00Z",
      "categories": ["standing", "frontal", "dynamic"],
      "tags": ["arms_raised", "looking_up"],
      "person_count": 1,
      "confidence": 0.94,
      "user_notes": "posa eroica, buona per copertine",
      "favorite": false
    }
  }
}
```

## Categorizzazione automatica

All'import, ogni posa viene analizzata e categorizzata automaticamente
(l'utente può poi correggere). Le categorie derivano dall'analisi dello
scheletro:

**Orientamento**: frontal, three-quarter, profile, back
**Postura**: standing, sitting, kneeling, lying, crouching, jumping
**Dinamica**: static, dynamic, action
**Arti**: arms_raised, arms_crossed, pointing, hand_on_hip, ...
**Numero persone**: single, couple, group

La categorizzazione usa euristiche sui joint estratti da DWPose
(es. se i polsi sono sopra le spalle → "arms_raised"). Non è perfetta
ma dà una base ricercabile; l'utente raffina.

## UI Pose Library

```
┌──────────────────────────────────────────────────┐
│  Pose Library                          [+ Importa] │
├──────────────────────────────────────────────────┤
│  Cerca: [____________]                             │
│  Filtri: [Standing ▼][Frontal ▼][Dynamic ▼][★]    │
├──────────────────────────────────────────────────┤
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│  │ 🖼  │ │ 🖼  │ │ 🖼  │ │ 🖼  │ │ 🖼  │ │ 🖼  │       │
│  │skel│ │skel│ │skel│ │skel│ │skel│ │skel│       │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘       │
│  ... griglia scrollabile con 500 pose ...          │
├──────────────────────────────────────────────────┤
│  Selezione: pose_0042                              │
│  [ Usa in "Personaggio in posa" ]  [ Modifica tag ]│
└──────────────────────────────────────────────────┘
```

Ogni cella mostra (toggle) o l'immagine originale o lo scheletro
estratto sovrapposto. Doppio click → dettaglio. La griglia gestisce 500+
immagini con lazy loading delle thumbnail (non carica tutto in RAM).

## Import workflow

```
1. Utente trascina N immagini / una cartella
2. Per ognuna:
   a. Copia in sources/ con nome pose_NNNN
   b. DWPose extraction → skeleton json + preview png
   c. Genera thumbnail 256px
   d. Categorizzazione automatica
   e. Entry in library.json
3. Progress bar (500 immagini ≈ 3-5 min su GPU)
4. Le pose dove DWPose non trova scheletro (confidence bassa) vengono
   segnalate: "12 immagini senza posa rilevabile, rivedi"
```

## Estrazione scheletro (src/pose/extractor.py)

Usa **DWPose** (più accurato di OpenPose classico):

```python
class PoseExtractor:
    def extract(self, image_path: Path) -> PoseData:
        """Restituisce joints, bones, confidence + preview renderizzata."""

    def render_skeleton(self, pose_data: PoseData, size: tuple) -> Image:
        """Rendering OpenPose-style per ControlNet input + preview UI."""
```

Lo scheletro estratto serve due scopi:
1. **Preview** per l'utente (vede la posa estratta)
2. **Input diretto ControlNet** (skip ri-estrazione a ogni generazione →
   più veloce)

## Uso in Ricetta A

Quando l'utente sceglie una posa dalla library per "Personaggio in
posa", l'app NON ri-estrae lo scheletro (è già in cache). Passa
direttamente lo skeleton png a ControlNet OpenPose. Risparmio di tempo
e VRAM a ogni generazione.

## Gestione collezione grande

500+ immagini richiedono attenzione alle performance:
- **Thumbnail lazy load**: la griglia carica thumbnail solo per le celle
  visibili (virtual scrolling)
- **library.json indicizzato**: ricerca/filtro in memoria su metadata,
  non rilettura file
- **Skeleton cache**: estrazione una volta sola all'import
- **Categorie indicizzate**: filtri istantanei

## Possibili evoluzioni

- **Pose editor**: aggiustare manualmente lo scheletro estratto
  (trascinare un joint) per pose perfette
- **Pose blending**: combinare due pose
- **Pose da testo**: generare uno scheletro da descrizione testuale
- **3D pose**: integrare un manichino 3D (come in Magic Poser) per
  creare pose custom da zero
