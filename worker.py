import time
import requests
import datetime
import pytz

API_URL = "https://ventas-sqlite-api.onrender.com/refresh_db"
TZ = pytz.timezone("America/Bogota")
HORA_EJECUCION = 3

def refrescar():
    ahora = datetime.datetime.now(TZ)
    print(f"[{ahora}] Ejecutando POST a: {API_URL}")

    try:
        r = requests.post(API_URL, timeout=300)
        print(f"[{ahora}] STATUS {r.status_code}")
        print("Respuesta del servidor:")
        print(r.text[:500])  # solo primeros 500 caracteres
    except Exception as e:
        print(f"[{ahora}] ERROR al llamar refresh_db: {e}")


def segundos_hasta_proxima_ejecucion():
    ahora = datetime.datetime.now(TZ)
    proximo = ahora.replace(hour=HORA_EJECUCION, minute=0, second=0, microsecond=0)

    if ahora >= proximo:
        proximo += datetime.timedelta(days=1)

    diff = (proximo - ahora).total_seconds()
    print(f"Próxima ejecución a las 3 AM: {proximo} (en {diff} seg)")
    return diff


if __name__ == "__main__":
    refrescar()

    while True:
        time.sleep(segundos_hasta_proxima_ejecucion())
        refrescar()
