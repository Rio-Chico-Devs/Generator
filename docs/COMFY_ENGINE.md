# ComfyUI Engine

Il cuore tecnico di Vihente Forge. Spiega come ComfyUI viene embeddato,
controllato e usato come motore di esecuzione, restando completamente
offline e invisibile all'utente.

## Perché ComfyUI e non diffusers

`diffusers` esegue una singola pipeline: prompt → immagine. Le ricette
di Vihente Forge richiedono di concatenare più modelli in sequenza
(estrai posa → applica ControlNet → applica IP-Adapter → applica LoRA →
auto-fix → upscale). Implementare questo a mano con diffusers è
possibile ma fragile e difficile da mantenere.

ComfyUI è progettato esattamente per questo: pipeline a grafo dove ogni
nodo è un'operazione (carica modello, applica ControlNet, campiona,
decodifica VAE, upscale...). I workflow sono file JSON serializzabili,
versionabili, parametrizzabili.

**Punto cruciale sull'indipendenza:** ComfyUI è software che gira sul
computer di Bru. Non è un servizio cloud. Una volta installato, non
contatta nessun server esterno (a meno di non scaricare nuovi nodi, che
è un'azione esplicita dell'utente, non automatica). È offline come tutto
il resto.

## Modello a due processi

```
Processo 1: Vihente Forge (PyQt6)
    │
    │  avvia all'avvio app, uccide alla chiusura
    ▼
Processo 2: ComfyUI headless
    - python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch
    - bind SOLO su localhost (mai esposto in rete)
    - nessuna UI web aperta (--disable-auto-launch)
```

L'app principale parla con ComfyUI via:
- **HTTP REST** per submit workflow e fetch risultati
- **WebSocket** per progress real-time durante l'esecuzione

## Lifecycle del processo ComfyUI

Gestito da `src/comfy/server.py`:

```
[AVVIO APP]
  1. Trova porta libera (default 8188, fallback incrementale)
  2. Lancia subprocess ComfyUI con flag corretti
  3. Poll health endpoint finché risponde (timeout 60s)
  4. Se non risponde: errore chiaro all'utente, app degrada
     a sola gestione progetti/dataset (no generazione)

[DURANTE USO]
  - Health check periodico (ogni 30s)
  - Se ComfyUI muore: tentativo restart automatico (max 3)
  - Logging stdout/stderr del subprocess in file dedicato

[CHIUSURA APP]
  1. SIGTERM al subprocess ComfyUI
  2. Attesa graceful (10s)
  3. SIGKILL se non termina
  4. Cleanup eventuali file temp
```

**Importante:** ComfyUI viene ucciso SEMPRE alla chiusura. Mai lasciare
processi orfani che occupano VRAM. Su crash dell'app, un watchdog
verifica all'avvio successivo se c'è un ComfyUI orfano sulla porta e lo
termina.

## Bundling: dove vive ComfyUI

```
~/Documents/Vihente Forge/
└── engine/
    └── ComfyUI/              ← installazione ComfyUI completa
        ├── main.py
        ├── custom_nodes/     ← nodi necessari per le ricette
        │   ├── ComfyUI-Impact-Pack/      (Adetailer)
        │   ├── ComfyUI_IPAdapter_plus/
        │   ├── comfyui_controlnet_aux/   (estrazione posa)
        │   ├── ComfyUI-IC-Light/
        │   └── ...
        └── models/           ← cartella nativa ComfyUI
```

I checkpoint/LoRA/VAE "ufficiali" di Vihente Forge vivono in
`~/Documents/Vihente Forge/models/{checkpoints,loras,vae}/`. ComfyUI li
vede SENZA symlink: `ComfyServer.start()` genera al volo un
`extra_model_paths.yaml` (root aggiuntiva, non sostitutiva) e lo passa
con `--extra-model-paths-config`. Niente junction/symlink: funziona
senza privilegi admin su Windows e non tocca la cartella nativa
`ComfyUI/models/`, dove restano eventuali modelli piazzati manualmente.

I custom_nodes necessari sono installati al primo avvio (download una
tantum da GitHub) oppure inclusi nel bundle PyInstaller per
installazione 100% offline. La versione di ComfyUI e dei nodi è
**pinnata** (commit hash specifici noti-funzionanti) per evitare che un
aggiornamento rompa le ricette.

## Client API (src/comfy/client.py)

Interfaccia Python per parlare con ComfyUI:

```python
class ComfyClient:
    def __init__(self, host="127.0.0.1", port=8188): ...

    def is_alive(self) -> bool:
        """Health check via GET /system_stats."""

    def submit(self, workflow: dict) -> str:
        """POST /prompt → restituisce prompt_id."""

    def wait_for_completion(
        self, prompt_id: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        """Ascolta WebSocket, chiama callback su ogni step,
        restituisce path delle immagini prodotte."""

    def interrupt(self) -> None:
        """POST /interrupt — ferma la generazione corrente."""

    def get_vram_usage(self) -> dict:
        """GET /system_stats → VRAM info per UI status bar."""
```

## Workflow parametrizzati (src/comfy/workflow.py)

I file in `assets/workflows/*.json` sono template ComfyUI con
placeholder. La nostra UI li carica, sostituisce i valori (prompt,
percorsi immagini, peso LoRA, seed...) e li invia.

```python
class WorkflowTemplate:
    def __init__(self, path: Path):
        self.graph = json.loads(path.read_text())

    def set_param(self, node_id: str, field: str, value): ...

    def set_lora(self, lora_path: str, weight: float): ...

    def set_input_image(self, node_id: str, image_path: str): ...

    def set_seed(self, seed: int): ...

    def build(self) -> dict:
        """Restituisce il workflow pronto per submit."""
```

I node_id e i campi sono mappati in una sezione del JSON o in un file
di mapping affiancato, così se cambia il workflow basta aggiornare il
mapping senza toccare il codice.

## Gestione VRAM in pipeline composite

ComfyUI ha gestione VRAM intelligente built-in:
- Carica in VRAM solo i modelli del nodo in esecuzione
- Scarica automaticamente quelli non più necessari nel grafo
- Su low-VRAM, usa `--lowvram` o `--normalvram` per offload aggressivo

Flag di avvio per 8GB:
```
--normalvram          # bilanciato (default per 8GB)
# oppure --lowvram    # se OOM persistente, più lento
--use-pytorch-cross-attention   # attention efficiente
```

Per le ricette più pesanti (Ricetta A con ControlNet + IP-Adapter +
LoRA + Adetailer), il picco è gestito da ComfyUI sequenziando i nodi:
non tutti i modelli sono in VRAM contemporaneamente.

## Sicurezza e isolamento

- ComfyUI bind solo su `127.0.0.1` — mai raggiungibile da rete esterna
- Porta scelta dinamicamente, nota solo all'app
- Nessun `--enable-cors-header` (no accesso da browser esterni)
- Custom nodes installati solo da lista whitelist nota (no arbitrary
  node execution da fonti non verificate)
- I workflow JSON sono nostri (in assets/), non scaricati a runtime
- Modelli solo safetensors verificati

## Modalità mock (sviluppo senza GPU)

`src/comfy/client.py` ha un `MockComfyClient` che:
- Non avvia processo reale
- Su submit, ritorna immagini placeholder dopo delay simulato
- Emette progress fake per testare la UI
- Permette sviluppo completo dell'interfaccia senza modelli/GPU

Attivato con flag `--mock`.

## Aggiornamenti ComfyUI

Strategia conservativa:
- Versione pinnata a commit noto-funzionante
- Aggiornamento è azione manuale (Settings → "Aggiorna engine")
- Prima dell'aggiornamento, backup della versione corrente
- Dopo aggiornamento, smoke test automatico (esegue workflow base)
- Rollback se lo smoke test fallisce

Mai aggiornamento automatico silenzioso: troppo rischio di rompere le
ricette in produzione.
