import time
import requests
import datetime
import pytz

API_URL = "https://ventas-sqlite-api.onrender.com/refresh_db"
TZ = pytz.timezone("America/Bogota")
HORA_EJECUCION = 3  # 3:00 AM

def refrescar():
    ahora = datetime.datetime.now(TZ)
    print(f"\n🔄 EXEC → Ejecutando POST a {API_URL} — {ahora}")

    try:
        r = requests.post(API_URL, timeout=300)
        print(f"📌 STATUS: {r.status_code}")
        print(f"📄 RESPUESTA (primeros 400 chars):\n{r.text[:400]}")
    except Exception as e:
        print(f"❌ ERROR ejecutando refresh_db:\n{e}")

def segundos_para_las_3am():
    ahora = datetime.datetime.now(TZ)
    run = ahora.replace(hour=HORA_EJECUCION, minute=0, second=0, microsecond=0)

    if ahora >= run:
        run += datetime.timedelta(days=1)

    diff = int((run - ahora).total_seconds())
    print(f"⏳ Próxima ejecución programada: {run} (en {diff} segundos)")
    return diff


if __name__ == "__main__":
    refrescar()  # se ejecuta al iniciar el worker

    while True:
        time.sleep(segundos_para_las_3am())
        refrescar()
