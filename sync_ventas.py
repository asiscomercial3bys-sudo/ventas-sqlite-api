import os
import csv
import io
from datetime import datetime

import requests
import psycopg2
from psycopg2.extras import execute_values

# Variables de entorno
DATABASE_URL = os.environ["DATABASE_URL"]   # Neon
CSV_URL = os.environ["CSV_URL"]             # link directo al CSV de ventas


def to_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def to_num(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    # Por si trae separador de miles o coma decimal (ej: 1.234,56)
    value = value.replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def to_date(value):
    """
    Convierte la fecha del CSV a formato YYYY-MM-DD.
    Intenta primero dd/mm/aaaa y luego aaaa-mm-dd.
    """
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None

    # dd/mm/aaaa (muy común en Colombia)
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.date()  # psycopg2 lo envía como DATE
        except ValueError:
            continue
    # Si no pudo parsear, la dejamos como None
    return None


def download_csv(url: str) -> io.StringIO:
    # Timeout para que no se quede colgado
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # asumimos CSV en UTF-8
    return io.StringIO(resp.text, newline='')


def sync_ventas():
    # 1. Descargar CSV
    print("Descargando CSV desde:", CSV_URL)
    f = download_csv(CSV_URL)
    reader = csv.DictReader(f)

    rows = list(reader)
    if not rows:
        print("No hay filas en el archivo de ventas")
        return

    print(f"Filas leídas del CSV: {len(rows)}")

    valores = []
    for r in rows:
        valores.append((
            to_int(r.get("Año")),
            to_date(r.get("Fecha")),            # convertido a DATE
            r.get("Mes"),
            r.get("Identificación+Suc"),
            r.get("Identificación"),
            r.get("Digito"),
            r.get("Suc"),
            r.get("Nombre cliente"),
            r.get("Vendedor"),
            r.get("Cobrador"),
            r.get("Código producto"),
            r.get("Nombre producto"),
            r.get("Categoria"),
            r.get("Referencia fábrica"),
            r.get("Grupo inventario"),
            r.get("Portafolio"),
            r.get("Unidad de medida"),
            to_num(r.get("Cantidad vendida")),
            to_num(r.get("Valor bruto")),
            to_num(r.get("Descuento")),
            to_num(r.get("Subtotal")),
            to_num(r.get("Impuesto cargo")),
            to_num(r.get("Impuesto retención")),
            to_num(r.get("Total")),
        ))

    # 3. Conexión a Neon
    print("Conectando a Neon…")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # OPCIÓN: borramos y recargamos todo cada vez
    print("Vaciando tabla ventas…")
    cur.execute("TRUNCATE TABLE ventas;")

    insert_sql = """
        INSERT INTO ventas (
            anio,
            fecha,
            mes,
            identificacion_suc,
            identificacion,
            digito,
            suc,
            nombre_cliente,
            vendedor,
            cobrador,
            codigo_producto,
            nombre_producto,
            categoria,
            referencia_fabrica,
            grupo_inventario,
            portafolio,
            unidad_medida,
            cantidad_vendida,
            valor_bruto,
            descuento,
            subtotal,
            impuesto_cargo,
            impuesto_retencion,
            total
        ) VALUES %s
    """

    print("Insertando filas en Neon…")
    execute_values(cur, insert_sql, valores)

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Se insertaron {len(valores)} filas en la tabla ventas.")


if __name__ == "__main__":
    sync_ventas()
