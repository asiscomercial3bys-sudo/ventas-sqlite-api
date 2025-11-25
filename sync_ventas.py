import os
import csv
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Aumentar el límite de tamaño de campo del CSV
# (para evitar _csv.Error: campo mayor que el límite del campo (131072))
csv.field_size_limit(10**7)

# =========================
#  Variables de entorno
# =========================

DATABASE_URL = os.environ["DATABASE_URL"]   # URL de Neon (Postgres)
CSV_URL = os.environ["CSV_URL"]             # Link directo al CSV de OneDrive/SharePoint


# =========================
#  Helpers de conversión
# =========================

def to_date(value):
    """
    Convierte fechas DD/MM/YYYY a YYYY-MM-DD.
    Si ya viene en formato correcto, la devuelve igual.
    Devuelve None si no se puede interpretar.
    """
    if not value:
        return None
    value = str(value).strip()
    if value == "":
        return None

    # Intento 1: DD/MM/YYYY (formato típico de Excel en español)
    try:
        return datetime.strptime(value, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        pass

    # Intento 2: YYYY-MM-DD (ya en formato ISO)
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
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
    """
    Convierte textos como '1.234,56' o '1234.56' a float.
    Devuelve None si está vacío o no se puede convertir.
    """
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None

    # quitar separador de miles y normalizar coma a punto
    value = value.replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


# =========================
#  Lectura en streaming del CSV
# =========================

def iter_csv_rows(url: str):
    """
    Descarga el CSV de forma 'streaming' y devuelve un iterador de filas (dicts).
    NO carga todo el archivo en memoria.
    """
    print(f"Descargando CSV desde: {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    # Algunas veces el CSV tiene BOM o caracteres raros al inicio; los ignoramos.
    lines = (
        line.decode("utf-8-sig", errors="ignore")
        for line in resp.iter_lines(decode_unicode=False)
        if line  # saltar líneas vacías
    )

    # IMPORTANTE: el delimitador es ; porque tu Excel en español guarda así el CSV
    reader = csv.DictReader(lines, delimiter=";")
    print(f"Encabezados detectados: {reader.fieldnames}")
    return reader


# =========================
#  Sincronización con Neon
# =========================

def sync_ventas(batch_size: int = 1000):
    print("Conectando a la base de datos...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Opcional: vaciar tabla antes de cargar
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

    # Recorrer filas del CSV en streaming
    for row in iter_csv_rows(CSV_URL):
        # OJO: los nombres deben coincidir con los encabezados reales de tu CSV
        valores = (
            to_int(row.get("Año")),
            to_date(row.get("Fecha")),                 # ← aquí usamos to_date
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

        # Cuando el lote alcance batch_size, lo enviamos a Neon
        if len(batch) >= batch_size:
            execute_values(cur, insert_sql, batch)
            conn.commit()
            total_insertadas += len(batch)
            print(f"Lote insertado. Acumuladas: {total_insertadas} filas.")
            batch.clear()

    # Insertar el último lote si quedó algo pendiente
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
