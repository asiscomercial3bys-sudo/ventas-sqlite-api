import time
import requests
import datetime
import pytz

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
API_URL = "https://ventas-sqlite-api.onrender.com/refresh_db"
TZ = pytz.timezone("America/Bogota")   # Zona horaria Colombia
HORA_EJECUCION = 3                     # 3:00 a.m.

# -----------------------------
# Función que ejecuta el refresh
# -----------------------------
def refrescar():
    ahora = datetime.datetime.now(TZ)
    print(f"[{ahora}] Ejecutando /refresh_db ...")

    try:
        r = requests.post(API_URL, timeout=120)
        print(f"[{ahora}] Respuesta: {r.status_code} -> {r.text}")
    except Exception as e:
        print(f"[{ahora}] ERROR llamando a /refresh_db: {e}")


# -----------------------------
# Calcula segundos hasta las 3:00 a.m.
# -----------------------------
def segundos_hasta_proxima_ejecucion():
    ahora = datetime.datetime.now(TZ)

    # Próxima ejecución hoy a las 3am
    proximo = ahora.replace(hour=HORA_EJECUCION, minute=0, second=0, microsecond=0)

    # Si ya pasó la hora de ejecución, programar para mañana
    if ahora >= proximo:
        proximo = proximo + datetime.timedelta(days=1)

    diferencia = (proximo - ahora).total_seconds()
    print(f"Ahora: {ahora}, Próxima ejecución: {proximo}, Segundos: {diferencia}")

    return diferencia


# -----------------------------
# PROCESO PRINCIPAL DEL WORKER
# -----------------------------
if __name__ == "__main__":
    # Ejecutar inmediatamente al iniciar (opcional)
    refrescar()

    while True:
        secs = segundos_hasta_proxima_ejecucion()
        print(f"Durmiendo {secs} segundos hasta la próxima ejecución a las 3:00 am...")
        time.sleep(secs)

        # Ejecutar el refresh
        refrescar()
