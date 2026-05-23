# Modelli

Catalogo completo: modello base + tutti i modelli ausiliari necessari
per le ricette. Tutti girano in locale, tutti scaricati una tantum.

## Modello base (lo stile generale)

| Modello | Repo | Dim | Licenza | Uso |
|---|---|---|---|---|
| **Pony Diffusion V6 XL** | `AstraliteHeart/pony-diffusion-v6-xl` | 7 GB | Fair AI Public 1.0-SD | Default: character, illustration |
| Juggernaut XL v9 | `RunDiffusion/Juggernaut-XL-v9` | 7 GB | OpenRAIL++-M | Fotorealismo, prodotti RCS |
| Animagine XL 4.0 | `cagliostrolab/animagine-xl-4.0` | 7 GB | Fair AI Public 1.0-SD | Illustration alternativa |
| SD 1.5 | `runwayml/stable-diffusion-v1-5` | 4 GB | CreativeML OpenRAIL-M | Sprite, training rapido |

VAE fix obbligatorio per SDXL fp16: `madebyollin/sdxl-vae-fp16-fix` (335 MB, MIT)

## Modelli ausiliari (per le ricette)

| Modello | Funzione | Dim | Ricetta | Licenza |
|---|---|---|---|---|
| **ControlNet OpenPose SDXL** | Vincola la posa | ~2.5 GB | A | OpenRAIL |
| DWPose / OpenPose detector | Estrae scheletro da reference | ~400 MB | A | Apache/MIT |
| **IP-Adapter FaceID Plus v2** | Mantiene identità personaggio | ~1 GB | A | Apache 2.0 |
| IP-Adapter Plus SDXL | Style/reference injection | ~850 MB | A, B | Apache 2.0 |
| ControlNet Hand Refiner | Fix mani | ~2.5 GB | A | OpenRAIL |
| **BiRefNet** | Background removal preciso | ~900 MB | B | MIT |
| **IC-Light** | Re-illuminazione prodotti | ~1.7 GB | B | Apache 2.0 |
| ControlNet Reference | Preserva il resto (correzione) | (no model) | C | — |
| Inpainting capability | Built-in nel base SDXL | — | C | — |
| **Real-ESRGAN x4plus** | Upscale generico | ~65 MB | tutte | BSD |
| 4x-UltraSharp | Upscale dettaglio illustrazione | ~67 MB | tutte | open |
| Adetailer (face/hand YOLO) | Detection per auto-fix | ~50 MB | A, B | AGPL* |

*Nota AGPL su YOLO: i modelli di detection di Adetailer usano YOLO
(Ultralytics, AGPL). Per uso personale di Bru: nessun problema. Per
distribuzione commerciale dell'app: valutare alternative (es. modelli
detection MIT) o licenza Ultralytics enterprise. Documentato per Fase 7.

## Captioning opzionale (training ad alta fedeltà)

| Modello | Funzione | Dim | Licenza |
|---|---|---|---|
| Llama 3.2 Vision 11B | Caption descrittive del dataset | ~7 GB (Q4) | Llama 3.2 Community |
| WD-EVA02 Large Tagger v3 | Tag Danbooru-style | ~400 MB | Apache 2.0 |

WD-tagger è il default (veloce, coerente con workflow Tensor Art di Bru).
Llama Vision è opzionale per caption in linguaggio naturale quando si
vuole massima fedeltà concettuale.

## Vettoriali (Fase 6)

| Tool | Funzione | Tipo | Licenza |
|---|---|---|---|
| vtracer | Raster → SVG | Rust binary/lib | MIT |
| potrace | Raster → SVG (B/N) | C lib | GPL |

Non sono modelli AI: sono vectorizer deterministici. L'app genera raster
pulito (palette ridotta, contorni netti) poi vectorizza. Vedi note in
HANDOFF su limiti del vettoriale.

## Totale spazio disco modelli

Setup completo Ricetta A + base: **~15 GB**
Setup completo tutte le ricette: **~20-22 GB**
Più captioning Llama Vision: **+7 GB**

Download una tantum. Dopo, completamente offline.

## Strategia download e indipendenza

Tre modalità:
1. **First-run guidato**: l'app scarica per gruppi (base prima, poi
   ausiliari per ricetta attivata). L'utente non aspetta 22GB subito.
2. **Import manuale**: punta a modelli già scaricati (se Bru ha già
   roba da ComfyUI/A1111 esistente)
3. **Bundle offline**: per installazione su macchina senza internet,
   tutti i modelli su disco esterno, copiati in models/

Download via huggingface_hub (hash verificati, resume). Dopo successo:
`HF_HUB_OFFLINE=1` → l'app rifiuta connessioni di rete.

## Licenze — riepilogo per uso di Bru

Uso personale (creazione asset propri, RCS, Vihente): **tutto OK**.

Per eventuale distribuzione futura dell'app a terzi, attenzione a:
- YOLO/Ultralytics (AGPL) → sostituibile
- ComfyUI (GPL) → ok se l'app resta separata o si apre il codice
- Pony/Animagine (Fair AI) → ok, ma propaga restrizioni d'uso

Per output commerciale (asset venduti/usati da RCS): gli output dei
modelli elencati sono utilizzabili commercialmente. Pony e SDXL
permettono uso commerciale degli output generati.

## Catalogo runtime

`src/core/catalog.py` mantiene il registro con metadata, requisiti VRAM,
URL, dipendenze tra modelli (es. Pony richiede sdxl-vae-fix; Ricetta A
richiede ControlNet OpenPose + IP-Adapter FaceID). Il download manager
risolve le dipendenze automaticamente quando l'utente attiva una ricetta.
