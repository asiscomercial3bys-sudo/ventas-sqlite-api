# prestart_download.py
import os
import sqlite3
import sys
import gdown

DB_ID = "1N_pnmmvYo7xPJB0gT6Dhl1ITCERw4cVW"
DB_PATH = "ventas2025.sqlite"
URL = f"https://drive.google.com/uc?id={DB_ID}"

def download_db():
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 10000:
        print(f"[prestart] Descargando DB desde {URL} ...", flush=True)
        gdown.download(URL, DB_PATH, quiet=False, fuzzy=True)
        print("[prestart] Descarga completa.", flush=True)
    else:
        print("[prestart] Base de datos ya existe, no se descarga.", flush=True)

def verify_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        if not tables:
            raise ValueError("No se encontraron tablas en la base de datos.")
        print(f"[prestart] Verificación OK. Tablas encontradas: {[t[0] for t in tables]}", flush=True)
    except Exception as e:
        print(f"[prestart] ERROR: La base no es válida o está corrupta -> {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    download_db()
    verify_db()
