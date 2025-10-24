# ---------- imports ----------
from pathlib import Path
import sqlite3, re, unicodedata, os
from typing import Dict, List, Optional, Literal, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---------- rutas y seguridad ----------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ventas2025.sqlite"   # Ruta para Render y local

PUBLIC_MODE = os.getenv("PUBLIC_MODE", "1") == "1"
API_KEY = os.getenv("API_KEY")

def require_auth(authorization: str = Header(None)):
    if PUBLIC_MODE:
        return
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Falta Authorization: Bearer <token>")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(403, "Token inválido")

# ---------- configuración ----------
DEFAULT_TABLE_HINTS = ["ventas_2025", "ventas", "reporte", "hoja1"]

# ---------- app ----------
app = FastAPI(title="Ventas API (SQLite)", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- conexión y helpers ----------
def get_conn():
    if not DB_PATH.exists():
        raise HTTPException(500, f"No existe la base en {DB_PATH}")
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

def query_db(sql: str, params: List = []) -> List[Dict]:
    """Ejecuta una consulta SQL y devuelve una lista de diccionarios."""
    try:
        with get_conn() as conn:
            df = pd.read_sql(sql, conn, params=params)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(400, f"Error ejecutando consulta: {e}")

def normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\s_]+", "", s)
    s = (s.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ñ","n"))
    return s

def list_tables() -> List[str]:
    with get_conn() as conn:
        df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1", conn)
    return df["name"].tolist()

def pick_table() -> str:
    tabs = list_tables()
    if not tabs:
        raise HTTPException(500, "La base no contiene tablas.")
    normtabs = {normalize(t): t for t in tabs}
    for hint in DEFAULT_TABLE_HINTS:
        h = normalize(hint)
        if h in normtabs:
            return normtabs[h]
    return tabs[0]

TABLA = pick_table()

def map_columns(tbl: str) -> Dict[str, str]:
    with get_conn() as conn:
        info = pd.read_sql(f"PRAGMA table_info([{tbl}])", conn)
    names = info["name"].tolist()
    norm_map = {normalize(c): c for c in names}
    m: Dict[str, str] = {}
    m["Cliente"] = norm_map.get("cliente") or "Cliente"
    m["Identificacion"] = norm_map.get("identificacion")
    m["Fecha"] = norm_map.get("fecha")
    m["GrupoInventario"] = norm_map.get("grupoinventario")
    m["Categoria"] = norm_map.get("categoria")
    m["Producto"] = norm_map.get("producto")
    m["Cantidad"] = norm_map.get("cantidad")
    m["Subtotal"] = norm_map.get("subtotal")
    m["Portafolio"] = norm_map.get("portafolio") or m.get("GrupoInventario")
    return m

COLS = map_columns(TABLA)

# ---------- helpers ----------
def ensure_period(desde: str, hasta: str):
    for d in (desde, hasta):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            raise HTTPException(422, "Formato de fecha inválido. Usa YYYY-MM-DD.")
    if desde > hasta:
        raise HTTPException(422, "El rango de fechas es inválido (desde > hasta).")

def parse_number_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s.astype(float)
    x = s.astype(str).str.replace(r"[\$\s\*\#\(\)]", "", regex=True)
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce").fillna(0.0)

# ---------- modelos ----------
class TopItem(BaseModel):
    nombre: str
    valor: float

class ResumenCliente(BaseModel):
    cliente: str
    desde: str
    hasta: str
    total_valor_subtotal: float
    resumen: List[Dict[str, object]]

# ---------- health ----------
@app.get("/")
def home():
    return {"ok": True, "tabla": TABLA, "public": PUBLIC_MODE, "docs": "/docs"}

@app.get("/health")
def health():
    with get_conn() as conn:
        fec_col = COLS["Fecha"]
        dfc = pd.read_sql(f"SELECT count(*) as n FROM [{TABLA}] WHERE [{fec_col}] IS NOT NULL", conn)
    cnt = int(dfc.iloc[0]["n"])
    return {"ok": True, "tabla": TABLA, "cols": COLS, "rows_con_fecha": cnt}

# ---------- nuevo endpoint: resumen por cliente ----------
@app.get("/resumen_cliente_general", response_model=ResumenCliente)
def resumen_cliente_general(
    cliente: str = Query(..., min_length=3, description="Nombre del cliente"),
    desde: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$")
):
    """
    Devuelve el resumen del cliente agrupado por Portafolio, Grupo de Inventario y Categoría.
    Valores expresados en Subtotal ($).
    """
    ensure_period(desde, hasta)
    val_col = COLS["Subtotal"]
    cli_col = COLS["Cliente"]
    fec_col = COLS["Fecha"]
    port_col = COLS["Portafolio"]
    grp_col = COLS["GrupoInventario"]
    cat_col = COLS["Categoria"]

    sql = f"""
        SELECT [{port_col}] AS Portafolio,
               [{grp_col}] AS GrupoInventario,
               [{cat_col}] AS Categoria,
               SUM([{val_col}]) AS Subtotal
        FROM [{TABLA}]
        WHERE lower([{cli_col}]) LIKE ?
        AND date([{fec_col}]) BETWEEN ? AND ?
        GROUP BY [{port_col}], [{grp_col}], [{cat_col}]
        ORDER BY Subtotal DESC
    """
    params = [f"%{cliente.lower()}%", desde, hasta]

    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        raise HTTPException(404, f"No se encontraron ventas para '{cliente}' en el rango {desde} → {hasta}.")

    df["Subtotal"] = parse_number_series(df["Subtotal"])
    total = float(df["Subtotal"].sum())

    resumen = df.round(2).to_dict(orient="records")

    return ResumenCliente(
        cliente=cliente,
        desde=desde,
        hasta=hasta,
        total_valor_subtotal=total,
        resumen=resumen
    )

# ---------- informe texto simple ----------
@app.get("/informe_cliente_texto")
def informe_cliente_texto(
    cliente: str,
    desde: str,
    hasta: str
):
    """Versión texto del resumen por cliente."""
    r = resumen_cliente_general(cliente=cliente, desde=desde, hasta=hasta)
    lineas = [
        f"🧾 Informe de ventas — {r.cliente}",
        f"Periodo: {r.desde} → {r.hasta}",
        f"Total ventas (subtotal): ${r.total_valor_subtotal:,.2f}",
        "",
        "Detalle por Portafolio / Grupo / Categoría:"
    ]
    for row in r.resumen:
        lineas.append(f"  - {row['Portafolio']} / {row['GrupoInventario']} / {row['Categoria']}: ${row['Subtotal']:,.2f}")
    return {"informe": "\n".join(lineas)}
