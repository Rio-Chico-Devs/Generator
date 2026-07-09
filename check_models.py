import json, os, glob
files = glob.glob(r"D:\VihenteForge\models\checkpoints\*.safetensors") + glob.glob(r"D:\VihenteForge\engine\ComfyUI\models\checkpoints\*.safetensors")
for f in files:
    try:
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
        expected = 8 + n + end
        actual = os.path.getsize(f)
        status = "OK" if actual >= expected else "CORROTTO/TRONCATO"
        print(status, "|", os.path.basename(f), "| atteso:", expected, "reale:", actual)
    except Exception as e:
        print("ERRORE |", os.path.basename(f), "|", e)
