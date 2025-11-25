import os
import csv
import io
from datetime import datetime

import requests
import psycopg2
from psycopg2.extras import execute_values

# subir límite de tamaño de campo del CSV
csv.field_size_limit(10_000_000)  # 10 MB por celda, de sobra


# Variables de entorno
DATABASE_URL = os.environ["DATABASE_URL"]   # URL de Neon (Postgres)
CSV_URL = os.environ["CSV_URL"]             # Link directo al CSV


# ---- Helpers ----

def to_date(value):
    if not value:
        return None
    value = value.strip()
    if value == "":
        return None

    # DD/MM/YYYY
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


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

    value = value.replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def iter_csv_rows(url: str):
    print(f"Descargando CSV desde: {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    lines = (
        line.decode("utf-8-sig", errors="ignore")
        for line in resp.iter_lines(decode_unicode=False)
        if line
    )

    # tu archivo va con ;
    reader = csv.DictReader(lines, delimiter=";")
    print(f"Encabezados detectados: {reader.fieldnames}")
    return reader


def sync_ventas(batch_size: int = 1000):
    print("Conectando a la base de datos...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    print("Vaciando tabla ventas (TRUNCATE)...")
    cur.execute("TRUNCATE TABLE ventas;")
    conn.commit()

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

    total_insertadas = 0
    batch = []

    for row in iter_csv_rows(CSV_URL):
        valores = (
            to_int(row.get("Año")),
            to_date(row.get("Fecha")),
            row.get("Mes"),
            row.get("Identificación+Suc"),
            row.get("Identificación"),
            row.get("Digito"),
            row.get("Suc"),
            row.get("Nombre cliente"),
            row.get("Vendedor"),
            row.get("Cobrador"),
            row.get("Código producto"),
            row.get("Nombre producto"),
            row.get("Categoria"),
            row.get("Referencia fábrica"),
            row.get("Grupo inventario"),
            row.get("Portafolio"),
            row.get("Unidad de medida"),
            to_num(row.get("Cantidad vendida")),
            to_num(row.get("Valor bruto")),
            to_num(row.get("Descuento")),
            to_num(row.get("Subtotal")),
            to_num(row.get("Impuesto cargo")),
            to_num(row.get("Impuesto retención")),
            to_num(row.get("Total")),
        )

        batch.append(valores)

        if len(batch) >= batch_size:
            execute_values(cur, insert_sql, batch)
            conn.commit()
            total_insertadas += len(batch)
            print(f"Lote insertado. Acumuladas: {total_insertadas} filas.")
            batch.clear()

    if batch:
        execute_values(cur, insert_sql, batch)
        conn.commit()
        total_insertadas += len(batch)
        print(f"Último lote insertado. Total final: {total_insertadas} filas.")

    cur.close()
    conn.close()
    print(f"Sincronización completada. Filas insertadas: {total_insertadas}")


if __name__ == "__main__":
    sync_ventas()
