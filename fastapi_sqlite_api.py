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
DB_PATH = BASE_DIR / "ventas2025.sqlite"   # Ruta relativa (Render y local)

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

# ---------- config ----------
DEFAULT_TABLE_HINTS = [
    "comparativo_emp._2024_vs_2025",
    "ventas_2025", "ventas", "reporte", "hoja1"
]

# ---------- app ----------
app = FastAPI(title="Ventas API (SQLite)", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- util db ----------
def get_conn():
    if not DB_PATH.exists():
        raise HTTPException(500, f"No existe la base en {DB_PATH}")
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

# ✅ NUEVO HELPER
def query_db(sql: str, params: List = []) -> List[Dict]:
    """Ejecuta una consulta SQL y devuelve una lista de diccionarios."""
    try:
        with get_conn() as conn:
            df = pd.read_sql(sql, conn, params=params)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(400, f"Error ejecutando consulta: {e}")

# ---------- autodetección ----------
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
    return m

COLS = map_columns(TABLA)

# ---------- helpers ----------
def ensure_period(desde: str, hasta: str):
    for d in (desde, hasta):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            raise HTTPException(422, "Formato de fecha inválido. Usa YYYY-MM-DD.")
    if desde > hasta:
        raise HTTPException(422, "El rango de fechas es inválido (desde > hasta).")

def extract_digits(s: str) -> str:
    return re.sub(r"\D+", "", str(s))

def parse_number_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s.astype(float)
    x = s.astype(str).str.replace(r"[\$\s\*\#\(\)]", "", regex=True)
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce").fillna(0.0)

# ---------- MODELOS ----------
class TopItem(BaseModel):
    nombre: str
    valor: float

class ResumenCliente(BaseModel):
    cliente: str
    cliente_id: Optional[str] = None
    desde: str
    hasta: str
    total_unidades: float
    total_valor_subtotal: float
    ticket_promedio: float
    ventas_por_grupo: Dict[str, float]
    top_productos: List[TopItem]
    mensual_ventas: Dict[str, float]
    mensual_unidades: Dict[str, float]

# ---------- ROOT ----------
@app.get("/")
def home():
    return {"ok": True, "tabla": TABLA, "public": PUBLIC_MODE, "docs": "/docs"}

@app.get("/health")
def health():
    with get_conn() as conn:
        fec_col = COLS["Fecha"]
        dfc = pd.read_sql(f"SELECT count(*) as n FROM [{TABLA}] WHERE [{fec_col}] IS NOT NULL", conn)
    cnt = int(dfc.iloc[0]["n"])
    return {"ok": True, "rows_con_fecha": cnt, "cols": COLS}

# ---------- CONSULTA CLIENTE ----------
@app.get("/consulta_cliente", response_model=ResumenCliente)
def consulta_cliente(
    cliente: Optional[str] = Query(None),
    identificacion: Optional[str] = Query(None),
    desde: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    grupo_inventario: Optional[str] = None,
    categoria: Optional[str] = None
):
    ensure_period(desde, hasta)
    val_col = COLS.get("Subtotal")
    cli_col = COLS.get("Cliente")
    fec_col = COLS.get("Fecha")
    grp_col = COLS.get("GrupoInventario")
    cat_col = COLS.get("Categoria")

    where = [f"date([{fec_col}]) BETWEEN ? AND ?"]
    params = [desde, hasta]
    if cliente:
        where.append(f"lower([{cli_col}]) LIKE ?")
        params.append(f"%{cliente.lower()}%")
    if grupo_inventario:
        where.append(f"lower([{grp_col}]) LIKE ?")
        params.append(f"%{grupo_inventario.lower()}%")
    if categoria:
        where.append(f"lower([{cat_col}]) LIKE ?")
        params.append(f"%{categoria.lower()}%")

    sql = f"""
        SELECT [{cli_col}] AS Cliente, [{grp_col}] AS GrupoInventario,
               [{cat_col}] AS Categoria, [{COLS.get('Producto')}] AS Producto,
               SUM([{val_col}]) AS Subtotal
        FROM [{TABLA}]
        WHERE {' AND '.join(where)}
        GROUP BY [{cli_col}], [{grp_col}], [{cat_col}], [{COLS.get('Producto')}]
    """
    df = pd.read_sql(sql, get_conn(), params=params)
    if df.empty:
        raise HTTPException(404, "No se encontraron registros.")

    total_valor = df["Subtotal"].sum()
    total_unidades = len(df)
    ticket_prom = total_valor / total_unidades

    return ResumenCliente(
        cliente=cliente or "N/A",
        cliente_id=None,
        desde=desde,
        hasta=hasta,
        total_unidades=float(total_unidades),
        total_valor_subtotal=float(total_valor),
        ticket_promedio=float(ticket_prom),
        ventas_por_grupo=df.groupby("GrupoInventario")["Subtotal"].sum().to_dict(),
        top_productos=[
            TopItem(nombre=r["Producto"], valor=float(r["Subtotal"])) for _, r in df.nlargest(10, "Subtotal").iterrows()
        ],
        mensual_ventas={},
        mensual_unidades={}
    )

# ✅ NUEVO ENDPOINT GLOBAL
@app.get("/consulta_grupo", summary="Consulta global por grupo de inventario")
def consulta_grupo(
    grupo_inventario: str = Query(..., min_length=2),
    desde: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    categoria: Optional[str] = None,
):
    ensure_period(desde, hasta)
    val_col = COLS.get("Subtotal")
    grp_col = COLS.get("GrupoInventario")
    cat_col = COLS.get("Categoria")
    prod_col = COLS.get("Producto")
    fec_col = COLS.get("Fecha")

    if not all([val_col, grp_col, cat_col, prod_col]):
        raise HTTPException(500, "Estructura de columnas incompleta. Revisa COLS.")

    where = [f"date([{fec_col}]) BETWEEN ? AND ?", f"[{grp_col}] LIKE ?"]
    params = [desde, hasta, f"%{grupo_inventario}%"]
    if categoria:
        where.append(f"[{cat_col}] LIKE ?")
        params.append(f"%{categoria}%")

    query = f"""
        SELECT [{grp_col}] AS GrupoInventario,
               [{cat_col}] AS Categoria,
               [{prod_col}] AS Producto,
               SUM([{val_col}]) AS Subtotal
        FROM [{TABLA}]
        WHERE {' AND '.join(where)}
        GROUP BY [{grp_col}], [{cat_col}], [{prod_col}]
        ORDER BY Subtotal DESC
    """
    rows = query_db(query, params)
    if not rows:
        raise HTTPException(404, f"No se encontraron registros para el grupo '{grupo_inventario}'.")
    return {
        "grupo_inventario": grupo_inventario,
        "desde": desde,
        "hasta": hasta,
        "categoria": categoria or "TODAS",
        "total_registros": len(rows),
        "ventas": rows
    }

# ---------- resto de endpoints (tops, ventas, valores, etc.) ----------
# ✅ aquí mantienes todos los endpoints adicionales de tu versión extendida
# (no se modifican)

