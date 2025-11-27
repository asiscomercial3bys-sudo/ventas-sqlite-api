import time
import requests
import datetime

# 👉 Aquí va la URL de TU API en Render
API_URL = "https://ventas-sqlite-api.onrender.com/refresh_db"

def refrescar():
    ahora = datetime.datetime.now()
    print(f"[{ahora}] Iniciando llamada a /refresh_db ...")

    try:
        r = requests.post(API_URL, timeout=120)
        print(f"[{ahora}] Respuesta: {r.status_code} -> {r.text}")
    except Exception as e:
        print(f"[{ahora}] ERROR llamando a /refresh_db: {e}")

if __name__ == "__main__":
    # Ejecuta una vez al arrancar el worker
    refrescar()

    # Luego entra en un ciclo infinito
    while True:
        # Esperar 24 horas (86400 segundos)
        time.sleep(86400)
        refrescar()
