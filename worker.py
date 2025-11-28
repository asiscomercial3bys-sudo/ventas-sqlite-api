# worker.py
import os
import time
import datetime as dt
import requests

# URL de tu API en Render (web service)
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://ventas-sqlite-api.onrender.com"   # cámbiala si tu URL es distinta
)

# Offset horario para Colombia (UTC-5)
# Puedes cambiarlo con una env var TZ_OFFSET_HOURS si algún día lo necesitas.
TZ_OFFSET_HOURS = int(os.getenv("TZ_OFFSET_HOURS", "-5"))

REFRESH_ENDPOINT = "/refresh_db"
CHECK_INTERVAL_SECONDS = 60  # revisar cada 60 segundos


def now_local():
    """Devuelve la hora local (Colombia) calculada desde UTC."""
    return dt.datetime.utcnow() + dt.timedelta(hours=TZ_OFFSET_HOURS)


def run_refresh():
    """Llama al endpoint /refresh_db de la API."""
    url = f"{API_BASE_URL}{REFRESH_ENDPOINT}"
    ahora = now_local().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ahora}] Llamando a {url} ...", flush=True)

    try:
        resp = requests.post(url, timeout=600)
        print(f"[{ahora}] Respuesta {resp.status_code}: {resp.text[:500]}", flush=True)
    except Exception as e:
        print(f"[{ahora}] ERROR llamando /refresh_db: {e}", flush=True)


def main():
    """
    Bucle infinito:
      - Calcula hora local.
      - Si son >= 3:00 am y aún no se ha ejecutado hoy, llama /refresh_db.
      - Espera 60s y vuelve a revisar.
    """
    last_run_date = None  # fecha (YYYY-MM-DD) del último día que se ejecutó

    print("=== Worker de actualización automática iniciado ===", flush=True)
    print(f"API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"TZ_OFFSET_HOURS = {TZ_OFFSET_HOURS}", flush=True)

    while True:
        ahora = now_local()
        hoy = ahora.date()

        # Si todavía no hemos corrido hoy y ya son las 3:00 am (o más)
        if (last_run_date != hoy) and (ahora.hour >= 3):
            print(f"[{ahora}] Es >= 07:00 y aún no se ha ejecutado hoy. Lanzando /refresh_db...", flush=True)
            run_refresh()
            last_run_date = hoy
        else:
            # Logs muy breves para no llenar la consola
            print(f"[{ahora}] Worker vivo. Última ejecución: {last_run_date}", flush=True)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
