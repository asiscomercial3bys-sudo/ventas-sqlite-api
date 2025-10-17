# prestart_download.py
from pathlib import Path
import sqlite3
import sys
import gdown

FILE_ID = "1N_pnmmvYo7xPJB0gT6Dhl1ITCERw4cVW"  # tu ID de Drive
DB_PATH = Path(__file__).parent / "ventas2025.sqlite"

def download_db():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"[prestart] Descargando DB desde {url} -> {DB_PATH}", flush=True)
    gdown.download(url, str(DB_PATH), quiet=False, fuzzy=True)
    print("[prestart] Descarga completa.", flush=True)

def verify_sqlite():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchone()[0]
        print(f"[prestart] Verificación OK. Objetos en la DB: {n}", flush=True)
    except Exception as e:
        print(f"[prestart] ERROR: La DB no es válida: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    # Descarga si no existe o si es sospechosamente pequeña (<1 MB)
    if (not DB_PATH.exists()) or (DB_PATH.stat().st_size < 1_000_000):
        download_db()
    else:
        print("[prestart] DB ya presente, se omite descarga.", flush=True)

    verify_sqlite()
