import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv("DATABASE_URL")
CSV_URL = os.getenv("CSV_URL")

CHUNK_SIZE = 5000  # lee de a 5000 filas para no explotar RAM

def process_chunk(df, cursor):
    records = df.where(pd.notnull(df), None).values.tolist()
    
    columns = ",".join(df.columns)
    placeholders = "(" + ",".join(["%s"] * len(df.columns)) + ")"

    sql = f"INSERT INTO ventas ({columns}) VALUES %s"
    
    execute_values(cursor, sql, records)

def main():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("Descargando CSV...")
    
    chunk_iter = pd.read_csv(CSV_URL, sep=";", chunksize=CHUNK_SIZE, encoding="latin1")

    for chunk in chunk_iter:
        print(f"Procesando {len(chunk)} filas...")
        process_chunk(chunk, cursor)
        conn.commit()

    cursor.close()
    conn.close()
    print("Sincronización finalizada OK.")

if __name__ == "__main__":
    main()
