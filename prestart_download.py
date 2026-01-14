# prestart_download.py
import os
import sqlite3
import sys

DB_PATH = "ventas2025.sqlite"

def verify_db():
    try:
        if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 10000:
            raise ValueError(f"No existe la base local en {DB_PATH} o está vacía. Ejecuta /refresh_db para crearla.")

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
    verify_db()
