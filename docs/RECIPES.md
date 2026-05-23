# Ricette (Pipeline)

Le "ricette" sono le pipeline composite che fanno il lavoro vero. Ogni
ricetta è un workflow ComfyUI orchestrato, esposto all'utente come una
funzione semplice. Questo documento le spiega nel dettaglio tecnico.

Glossario rapido:
- **ControlNet** = vincola la generazione a una struttura (posa, contorni, profondità)
- **IP-Adapter** = "image prompt", inietta l'aspetto di un'immagine di riferimento
- **Adetailer** = ri-genera automaticamente zone problematiche (volti, mani) a risoluzione maggiore
- **Inpainting** = ri-dipinge solo una zona mascherata
- **LoRA/LyCORIS** = il file di stile addestrato di Bru

---

## Ricetta A — Personaggio in posa (PRIORITÀ 1)

### Obiettivo
Dato: una posa di riferimento + un'immagine del personaggio + lo stile
di Bru. Produrre: il personaggio nella posa esatta, nello stile di Bru,
con anatomia corretta.

### Input utente
1. Posa (da Pose Library o upload al volo) — foto, sketch, qualsiasi cosa
2. Personaggio di riferimento (1-3 immagini per consistenza facciale)
3. Prompt testuale (cosa fa, dove, atmosfera)
4. Progetto attivo (determina LoRA stile)

### Pipeline ComfyUI

```
┌─ Posa reference
│      ↓
│   DWPose Estimator
│      ↓ (scheletro: joints + bones)
│   ControlNet Apply (OpenPose model, weight 0.8-1.0)
│      ↓
├─ Personaggio reference
│      ↓
│   IP-Adapter FaceID Plus v2 (weight 0.6-0.8)
│      ↓
├─ Stile (LoRA Bru)
│      ↓
│   Load LoRA (weight 0.85)
│      ↓
└──→ Modello base (Pony V6 XL) + tutti i condizionamenti
       ↓
     KSampler (DPM++ 2M Karras, 30 step, CFG 7)
       ↓
     VAE Decode → immagine grezza
       ↓
     Adetailer face (rileva volto, re-inferenza ad alta risoluzione)
       ↓
     Adetailer hand (rileva mani, re-inferenza con hand-refiner)
       ↓
     [opzionale] Upscale 2x (Real-ESRGAN)
       ↓
     Output finale
```

### Parametri chiave e perché
- **ControlNet weight 0.8-1.0**: alto perché la posa deve essere fedele.
  Se troppo alto (>1.0) può "congelare" anche dettagli indesiderati.
- **IP-Adapter weight 0.6-0.8**: medio. Troppo alto e il personaggio
  domina sullo stile; troppo basso e perde identità.
- **LoRA weight 0.85**: leggermente sotto 1.0 per lasciare al modello
  base un margine di coerenza anatomica.
- **Adetailer**: cruciale. Senza, le mani sono il punto debole numero
  uno. Con hand-refiner ControlNet dedicato, drasticamente migliori.

### Difetti gestiti
- Mani deformi → Adetailer hand + hand-refiner ControlNet
- Volto incoerente tra generazioni → IP-Adapter FaceID
- Posa imprecisa → ControlNet weight tuning
- Stile diluito → LoRA weight + trigger tag nel prompt

### Fedeltà attesa
Posa: 85-92% | Personaggio: 80-88% | Stile: 90-95%

---

## Ricetta B — Prodotto in scena (PRIORITÀ 2)

### Obiettivo
Dato: foto di un prodotto reale + tipo di output desiderato. Produrre:
materiale grafico promozionale (volantino, post, banner) con il prodotto
integrato coerentemente, nello stile di Bru, con spazio per testo.

### Perché un solo prompt non basta
Chiedere a un modello "crea un volantino con questo prodotto" fallisce
sempre: il modello reinventa il prodotto invece di usarlo, sbaglia
proporzioni, ignora il brand. La soluzione è separare prodotto e scena.

### Pipeline ComfyUI

```
┌─ Foto prodotto (da pool prodotti)
│      ↓
│   BiRefNet Background Removal
│      ↓ (prodotto isolato, alpha channel)
│   [conservato per compositing finale]
│
├─ Generazione scena/background
│      ↓
│   txt2img con LoRA stile Bru
│   prompt: descrizione scena + "spazio centrale vuoto per prodotto"
│      ↓ (sfondo nello stile, composizione pianificata)
│
├─ Re-illuminazione prodotto
│      ↓
│   IC-Light: ri-illumina il prodotto isolato in base alla luce
│   della scena generata (coerenza luce/ombra)
│      ↓
│   Compositing: prodotto re-illuminato sopra la scena
│      ↓
├─ Layer testo
│      ↓
│   Generato come SVG overlay con font reali (NON AI text)
│   posizionato nello spazio pianificato
│      ↓
└──→ Final pass: img2img leggero (strength 0.2) per uniformare
       ↓
     Output: composizione completa, prodotto fedele, testo nitido
```

### Perché il testo è SVG e non AI
I modelli di diffusione sbagliano quasi sempre il testo (lettere
deformi, parole inventate). La soluzione professionale: il testo è un
layer vettoriale separato con font reali, sovrapposto in
post-processing. Nitido, editabile, corretto al 100%.

### Pool prodotti
Una libreria di immagini prodotto (foto RCS) gestita come la Pose
Library: cartella organizzata, preview, ricerca. Bru fotografa i
prodotti una volta, li importa, e li riusa per infiniti materiali.

### Template output
Preset dimensioni e layout: volantino A4/A5, post Instagram quadrato,
story verticale, hero banner web, OG image. Ogni template definisce
dove va il prodotto e dove va il testo.

### Fedeltà attesa
Prodotto: 95%+ (resta la foto reale, solo re-illuminata) | Layout: alto

---

## Ricetta C — Correggi / rifinisci (PRIORITÀ 3, trasversale)

### Obiettivo
Dato: un'immagine + zona da correggere. Produrre: la stessa immagine
con la zona sistemata, SENZA toccare il resto e SENZA aggiungere
elementi non richiesti.

### Il problema "l'AI aggiunge roba"
È un problema di approccio, non di tecnologia. Si risolve con tre
accorgimenti combinati.

### Pipeline ComfyUI

```
┌─ Immagine + mask (disegnata col brush nell'UI)
│      ↓
│   ControlNet Reference (preserva struttura/stile del resto)
│      ↓
│   Inpainting nella sola area mascherata
│   - strength bassa (0.4-0.6): modifica senza stravolgere
│   - denoise limitato: non inventa, corregge
│      ↓
│   [genera 4 varianti con seed diversi in parallelo]
│      ↓
└──→ UI mostra le 4 → utente sceglie → applica solo quella
```

### Perché funziona
- **Mask precisa**: solo la zona selezionata viene toccata, il resto è
  letteralmente copiato pixel per pixel dall'originale
- **ControlNet Reference**: anche dentro la mask, il modello "guarda"
  il resto dell'immagine per coerenza
- **Strength bassa**: la differenza tra correzione e re-invenzione. Bassa
  = sistema ciò che c'è. Alta = ridisegna da zero.
- **Multiple seed**: l'utente sceglie, non subisce il primo risultato

### Casi d'uso
- Sistemare una mano sbagliata in un'immagine altrimenti perfetta
- Rimuovere un elemento di sfondo indesiderato
- Modificare un dettaglio (colore di un oggetto, espressione)
- Completare una zona mancante/danneggiata

---

## Ricetta base — Genera nel mio stile

Il caso semplice, sempre disponibile come punto di partenza.

```
Prompt + LoRA stile Bru → txt2img → Adetailer → output
```

Per esplorazione, brainstorming visivo, o quando non serve né posa né
prodotto. È la ricetta più veloce (~15s).

---

## Come le ricette diventano UI

Ogni ricetta ha una view dedicata in `src/ui/recipe_views/`. La
complessità della pipeline è nascosta: l'utente vede solo gli input che
gli competono.

Esempio UI Ricetta A:
```
┌─────────────────────────────────────────┐
│  Personaggio in posa                     │
├─────────────────────────────────────────┤
│  [ Scegli posa ▼ ]  oppure  [ Carica ]  │
│   (anteprima posa selezionata)           │
│                                          │
│  [ Personaggio ▼ ]  (anteprima)         │
│                                          │
│  Descrizione: [_______________________]  │
│                                          │
│  Stile: Iris v3 (dal progetto attivo)   │
│                                          │
│  [ Avanzate ▾ ]   ← pesi, seed, etc.    │
│                                          │
│         [    Genera    ]                 │
└─────────────────────────────────────────┘
```

I parametri tecnici (ControlNet weight, IP-Adapter weight, Adetailer
on/off) stanno tutti sotto "Avanzate", con default sensati. L'utente
normale non li tocca mai.

---

## Estensione: nuove ricette future

Aggiungere una ricetta richiede:
1. Un workflow ComfyUI JSON in `assets/workflows/`
2. Una entry in `src/core/recipes.py` (input richiesti, mapping parametri)
3. Una view in `src/ui/recipe_views/`

Idee per ricette future:
- **Espandi immagine** (outpainting per cambiare aspect ratio)
- **Style transfer** (applica stile Bru a foto qualsiasi)
- **Character sheet** (stesso personaggio, multiple pose, in griglia)
- **Variazioni colore** (stessa immagine, palette diverse)
- **Da sketch a finito** (la tua matita → illustrazione completa)
