import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import sys
sys.path.insert(0, ".")
from pathlib import Path
from src.comfy.server import ComfyServer
from src.comfy.client import ComfyClient
from src.comfy.workflow import WorkflowTemplate

server = ComfyServer()
print("ComfyUI dir:", server.comfy_dir)
print("Installato:", server.is_installed())

port = server.start(timeout=90)
print("ComfyUI avviato su porta", port)

try:
    wf = WorkflowTemplate(Path("assets/workflows/base_txt2img.json"))
    wf.set_role("checkpoint", "WAI-Nsfw-Illustrious-17.safetensors")
    wf.set_loras([])  # nessun LoRA per questo primo test
    wf.set_prompt(
        "masterpiece, best quality, 1girl, solo, standing, looking at viewer",
        "worst quality, low quality, blurry",
    )
    wf.set_dimensions(1024, 1024)
    wf.set_seed(42)

    client = ComfyClient(port=port)
    prompt_id = client.submit(wf.build())
    print("Submit ok, prompt_id =", prompt_id)

    def on_progress(done, total):
        print(f"  step {done}/{total}")

    paths = client.wait_for_completion(prompt_id, progress_callback=on_progress)
    print("Immagini generate:")
    for p in paths:
        print(" -", p)
finally:
    server.stop()
    print("ComfyUI fermato")