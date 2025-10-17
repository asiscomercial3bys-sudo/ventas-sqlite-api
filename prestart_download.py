# prestart_download.py
from pathlib import Path
import os, sys, requests

DB_PATH = Path(os.getenv("DB_PATH", "ventas2025.sqlite"))
DB_URL = "https://drive.google.com/uc?export=download&id=1N_pnmmvYo7xPJB0gT6Dhl1ITCERw4cVW"

if DB_PATH.exists():
    print(f"[prestart] DB ya existe -> {DB_PATH.resolve()}")
    sys.exit(0)

if not DB_URL:
    print("[prestart] DB no existe y no hay DB_URL; continuo sin descargar.")
    sys.exit(0)

print(f"[prestart] Descargando DB desde {DB_URL} -> {DB_PATH} ...")
with requests.get(DB_URL, stream=True) as r:
    r.raise_for_status()
    with open(DB_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1<<20):
            if chunk:
                f.write(chunk)
print("[prestart] Descarga completa.")

