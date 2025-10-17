# ---------- imports ----------
from pathlib import Path
import sqlite3, re, unicodedata, os
from typing import Dict, List, Optional, Literal
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from pydantic import BaseModel

# ---------- rutas y seguridad ----------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ventas2025.sqlite"   # Ruta relativa (para Render y local)

API_KEY = os.getenv("API_KEY")  # en Render la defines
def require_auth(authorization: str = Header(None)):
    """
    Auth tipo Bearer. Si no hay API_KEY en el entorno (entorno local), no exige auth.
    """
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Falta Authorization: Bearer <token>")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(403, "Token inválido")

# ---------- config ----------
DEFAULT_TABLE_HINTS = [
    "comparativo_emp._2024_vs_2025",  # pista principal
    "ventas_2025", "ventas", "reporte", "hoja1"
]

# ---------- app ----------
app = FastAPI(title="Ventas API (SQLite)", version="1.0.0")

# ---------- util db ----------
def get_conn():
    if not DB_PATH.exists():
        raise HTTPException(500, f"No existe la base en {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))

def list_tables() -> List[str]:
    with get_conn() as conn:
        df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1", conn)
    return df["name"].tolist()

def normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\s_]+", "", s)
    s = (s.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ñ","n"))
    return s

def strip_accents(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace(" ", "")

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

def pick_fuzzy(norm_map: Dict[str,str], *cands) -> Optional[str]:
    for cand in cands:
        c = normalize(cand)
        for key, orig in norm_map.items():
            if c in key:
                return orig
    return None

def map_columns(tbl: str) -> Dict[str,str]:
    with get_conn() as conn:
        info = pd.read_sql(f"PRAGMA table_info([{tbl}])", conn)
    names = info["name"].tolist()
    norm_map = {normalize(c): c for c in names}

    m: Dict[str,str] = {}
    m["Cliente"] = (norm_map.get(normalize("Nombre cliente")) or
                    pick_fuzzy(norm_map, "nombre cliente", "cliente", "cliente/mes"))
    m["Fecha"] = (norm_map.get(normalize("Fecha")) or
                  pick_fuzzy(norm_map, "fecha", "fechaventa", "date"))
    m["Portafolio"] = (norm_map.get(normalize("Portafolio")) or
                       norm_map.get(normalize("Grupo inventario")) or
                       pick_fuzzy(norm_map, "portafolio", "grupoinventario", "categoria", "linea", "grupo"))
    m["Producto"] = (norm_map.get(normalize("Nombre producto")) or
                     pick_fuzzy(norm_map, "nombreproducto", "producto", "codigo", "codigoproducto"))
    m["Cantidad"] = (norm_map.get(normalize("Cantidad vendida")) or
                     pick_fuzzy(norm_map, "cantidadvendida", "cantidad", "unidades", "unid", "cant", "qty"))

    # Posibles columnas de valor
    m["Total"]      = norm_map.get(normalize("Total"))
    m["Subtotal"]   = norm_map.get(normalize("Subtotal"))
    m["ValorBruto"] = norm_map.get(normalize("Valor bruto"))
    m["Valor"] = (m["Total"] or m["Subtotal"] or m["ValorBruto"])

    missing = [k for k in ["Cliente","Fecha","Cantidad"] if not m.get(k)]
    if missing:
        raise HTTPException(400, f"Faltan columnas mínimas en [{tbl}]: {missing}. Revisa nombres.")
    return m

# ---------- modelos ----------
class TopItem(BaseModel):
    producto: str
    valor: float

class ResumenCliente(BaseModel):
    cliente: str
    desde: str
    hasta: str
    total_unidades: float
    total_valor: float
    ticket_promedio: float
    ventas_por_portafolio: Dict[str, float]
    top_productos: List[TopItem]

# ---------- autodetección ----------
TABLA = pick_table()
COLS  = map_columns(TABLA)

# ---------- endpoints básicos ----------
@app.get("/health")
def health():
    return {"ok": True, "db": str(DB_PATH), "tabla": TABLA, "cols": COLS}

@app.get("/tablas")
def tablas():
    return {"tablas": list_tables()}

@app.get("/schema")
def schema():
    with get_conn() as conn:
        df = pd.read_sql(f"PRAGMA table_info([{TABLA}])", conn)
    return {"tabla": TABLA, "columns": df.to_dict(orient="records")}

# ---------- consulta cliente ----------
@app.get("/consulta_cliente", response_model=ResumenCliente, dependencies=[Depends(require_auth)])
def consulta_cliente(
    cliente: str = Query(..., min_length=2, description="Nombre o parte del nombre"),
    desde: str   = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str   = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    monto: Literal["total","subtotal","bruto"] = Query("total", description="Campo de monto a usar")
):
    # Variables de columnas
    cli_col = COLS["Cliente"]; fec_col = COLS["Fecha"]; por_col = COLS["Portafolio"]
    pro_col = COLS["Producto"]; qty_col = COLS["Cantidad"]

    # Selección de columna de valor
    if monto == "subtotal" and COLS.get("Subtotal"):
        val_col = COLS["Subtotal"]
    elif monto == "bruto" and COLS.get("ValorBruto"):
        val_col = COLS["ValorBruto"]
    else:
        val_col = COLS.get("Valor")  # fallback

    # Consulta
    with get_conn() as conn:
        sql = f"""
            SELECT
                [{cli_col}]  AS Cliente,
                [{fec_col}]  AS Fecha,
                [{por_col}]  AS Portafolio,
                [{pro_col}]  AS Producto,
                CAST([{qty_col}] AS FLOAT)  AS Cantidad
                {"," if val_col else ""} {f"CAST([{val_col}] AS FLOAT) AS Valor" if val_col else ""}
            FROM [{TABLA}]
            WHERE date([{fec_col}]) BETWEEN ? AND ?
              AND lower([{cli_col}]) LIKE lower(?)
        """
        df = pd.read_sql(sql, conn, params=[desde, hasta, f"%{cliente}%"])

    if df.empty:
        raise HTTPException(404, "Sin datos para ese filtro.")

    if "Valor" not in df.columns:
        df["Valor"] = 0.0

    # Limpieza numérica
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0.0)
    df["Valor"]    = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)

    # Filtrar al cliente exacto (evitar mezclar homónimos)
    cliente_det = df["Cliente"].value_counts().idxmax()
    df_cli = df[df["Cliente"] == cliente_det].copy()

    # Cálculos
    total_unidades = float(df_cli["Cantidad"].sum())
    total_valor    = float(df_cli["Valor"].sum())
    ticket_prom    = float(total_valor / total_unidades) if total_unidades else 0.0

    # Agrupaciones
    por_porta = (
        df_cli.groupby("Portafolio", dropna=False)["Valor"]
              .sum().sort_values(ascending=False).round(2).to_dict()
    )
    top_prod_df = (
        df_cli.groupby("Producto", dropna=False)["Valor"]
              .sum().sort_values(ascending=False).head(10).reset_index()
              .rename(columns={"Valor": "valor"})
    )
    top_list = [
        TopItem(
            producto=("N/A" if pd.isna(r["Producto"]) else str(r["Producto"])),
            valor=float(r["valor"])
        )
        for _, r in top_prod_df.iterrows()
    ]

    return ResumenCliente(
        cliente=str(cliente_det),
        desde=desde,
        hasta=hasta,
        total_unidades=round(total_unidades, 2),
        total_valor=round(total_valor, 2),
        ticket_promedio=round(ticket_prom, 2),
        ventas_por_portafolio={ (k if k is not None else "N/A"): float(v) for k,v in por_porta.items() },
        top_productos=top_list
    )

# ---------- informe (formato amigable) ----------
@app.get("/informe_cliente", dependencies=[Depends(require_auth)])
def informe_cliente(
    cliente: str,
    desde: str,
    hasta: str,
    monto: Literal["total","subtotal","bruto"] = "total"
):
    try:
        r = consulta_cliente(cliente=cliente, desde=desde, hasta=hasta, monto=monto)
    except HTTPException as e:
        if e.status_code == 404:
            return {"informe": f"No encontré ventas para '{cliente}' en {desde} → {hasta}."}
        raise

    lineas = [
        f"🧾 Informe de cliente: {r.cliente}",
        f"Periodo: {r.desde} → {r.hasta}",
        f"Unidades: {r.total_unidades:,.0f}",
        f"Ventas: ${r.total_valor:,.2f}",
        f"Ticket promedio: ${r.ticket_promedio:,.2f}",
        "",
        "Portafolio (Top):"
    ]
    port = sorted(r.ventas_por_portafolio.items(), key=lambda x: x[1], reverse=True)
    for k, v in port[:10]:
        lineas.append(f"  - {k}: ${v:,.2f}")

    lineas.append("")
    lineas.append("Top productos:")
    for i, tp in enumerate(r.top_productos[:10], 1):
        lineas.append(f"  {i}. {tp.producto} — ${tp.valor:,.2f}")

    return {"informe": "\n".join(lineas)}
