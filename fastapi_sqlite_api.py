# ---------- imports ----------
from pathlib import Path
import sqlite3, re, unicodedata, os
from typing import Dict, List, Optional, Literal, Tuple

import io
import requests

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---------- rutas y seguridad ----------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ventas2025.sqlite"   # Ruta relativa (Render y local)

# CSV remoto (OneDrive) para recargar la base
CSV_URL = os.getenv("CSV_URL")  # Link directo de descarga al CSV (termina en ?download=1)

# Modo público: si es "1" no exige token desde ningún cliente
PUBLIC_MODE = os.getenv("PUBLIC_MODE", "1") == "1"
API_KEY = os.getenv("API_KEY")  # solo se usa si PUBLIC_MODE=False

def require_auth(authorization: str = Header(None)):
    """
    Si PUBLIC_MODE es True => no exige auth.
    Si PUBLIC_MODE es False => exige encabezado Bearer y que coincida con API_KEY.
    """
    if PUBLIC_MODE:
        return
    if not API_KEY:
        return  # sin API_KEY definida, no se exige auth
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
app = FastAPI(title="Ventas API (SQLite)", version="3.0.2")

# CORS abierto
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
    # Evitar bloqueos con Uvicorn
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

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

def strip_accents_spaces_lower(s: str) -> str:
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
    # Cliente (nombre)
    m["Cliente"] = (norm_map.get(normalize("Nombre cliente")) or
                    pick_fuzzy(norm_map, "nombre cliente", "cliente", "cliente/mes"))

    # Identificación
    m["Identificacion"] = (
        norm_map.get(normalize("Identificación")) or
        norm_map.get(normalize("Identificacion"))  or
        pick_fuzzy(
            norm_map,
            "identificacion+suc", "identificacion", "nit", "nitcliente", "rut", "ruc",
            "dni", "cedula", "cedulacliente", "doc", "documento", "numdocumento",
            "idcliente", "clienteid", "nrodoc", "numero documento", "no documento"
        )
    )

    # Fecha
    m["Fecha"] = (norm_map.get(normalize("Fecha")) or
                  pick_fuzzy(norm_map, "fecha", "fechaventa", "date"))

    # Grupo de inventario / Portafolio
    m["GrupoInventario"] = (norm_map.get(normalize("Grupo inventario")) or
                            pick_fuzzy(norm_map, "grupoinventario", "grupo", "linea", "categoria", "portafolio"))

    # Portafolio (compat)
    m["Portafolio"] = (norm_map.get(normalize("Portafolio")) or m["GrupoInventario"])

    # Categoria
    m["Categoria"] = (
        norm_map.get(normalize("Categoria")) or
        pick_fuzzy(norm_map, "categoria", "subcategoria", "familia", "rubro", "clase", "segmento")
    )

    # Producto
    m["Producto"] = (norm_map.get(normalize("Nombre producto")) or
                     pick_fuzzy(norm_map, "nombreproducto", "producto", "codigo", "codigoproducto"))

    # Cantidad
    m["Cantidad"] = (norm_map.get(normalize("Cantidad vendida")) or
                     pick_fuzzy(norm_map, "cantidadvendida", "cantidad", "unidades", "unid", "cant", "qty"))

    # Subtotal (monto)
    m["Subtotal"] = norm_map.get(normalize("Subtotal"))

    missing = [k for k in ["Cliente","Fecha","Cantidad","Subtotal"] if not m.get(k)]
    if missing:
        raise HTTPException(400, f"Faltan columnas mínimas en [{tbl}]: {missing}. Revisa nombres.")
    return m

# ---------- autodetección ----------
TABLA = pick_table()
COLS  = map_columns(TABLA)

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
    """
    Convierte texto numérico con formatos LATAM/EN a float y limpia símbolos.
    """
    if s.empty:
        return s.astype(float)
    x = s.astype(str).str.replace(r"[\$\s\*\#\(\)]", "", regex=True)
    latam_mask = x.str.contains(",", na=False) & x.str.contains(r"\.", na=False)
    x_latam = x[latam_mask].str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
    only_comma_mask = x.str.contains(",", na=False) & ~x.str.contains(r"\.", na=False)
    x_onlyc = x[only_comma_mask].str.replace(",", ".", regex=False)
    rest_mask = ~(latam_mask | only_comma_mask)
    x_rest = x[rest_mask].str.replace(",", "", regex=True)
    x2 = pd.concat([x_latam, x_onlyc, x_rest]).reindex(x.index)
    return pd.to_numeric(x2, errors="coerce").fillna(0.0)

# --- Normalizaciones robustas para filtros de texto (SQL y Python) ---
def sql_norm_column(col: str) -> str:
    """
    Normaliza una columna en SQL: minúsculas, sin espacios, signos ni acentos (LATAM).
    OJO: comillas se limpian con char(39)/char(34) para no romper el SQL.
    """
    x = f"lower([{col}])"
    x = f"replace({x}, char(160), '')"  # NBSP
    x = f"replace({x}, ' ', '')"
    # signos seguros como literales
    for ch in [".", ",", "-", "/", "+", "_", "(", ")"]:
        x = f"replace({x}, '{ch}', '')"
    # comilla simple y doble con char() para evitar "unrecognized token"
    x = f"replace({x}, char(39), '')"  # '
    x = f"replace({x}, char(34), '')"  # "
    # acentos y eñes
    for a,b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),
                ("à","a"),("è","e"),("ì","i"),("ò","o"),("ù","u"),
                ("ñ","n")]:
        x = f"replace({x}, '{a}', '{b}')"
    return x

def py_norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[.,\\-\\/\\+_()'\"“”’]", "", s)
    s = s.replace("\u00A0","")
    return s

def add_text_filter(where: List[str], params: List[str], value: Optional[str], col: Optional[str]):
    if value and col:
        norm_col = sql_norm_column(col)
        norm_val = py_norm_text(value)
        where.append(f"({norm_col} = ? OR {norm_col} LIKE ?)")
        params.extend([norm_val, f"%{norm_val}%"])

# ---------- helpers SQL ----------
def build_base_select(val_col: str, extra_where: str = "", select_cols: Optional[List[str]] = None) -> Tuple[str, List]:
    cli_col = COLS["Cliente"]; fec_col = COLS["Fecha"]
    pro_col = COLS["Producto"]; grp_col = COLS.get("GrupoInventario")
    cat_col = COLS.get("Categoria")
    qty_col = COLS["Cantidad"]

    sel = [
        f"[{cli_col}]  AS Cliente",
        f"[{fec_col}]  AS Fecha",
        f"[{pro_col}]  AS Producto",
        f"[{qty_col}]  AS Cantidad",
        f"[{val_col}]  AS Subtotal",
    ]
    insert_pos = 2
    if grp_col:
        sel.insert(insert_pos, f"[{grp_col}] AS GrupoInventario"); insert_pos += 1
    else:
        sel.insert(insert_pos, "'N/A' AS GrupoInventario"); insert_pos += 1
    if cat_col:
        sel.insert(insert_pos, f"[{cat_col}] AS Categoria")
    else:
        sel.insert(insert_pos, "'N/A' AS Categoria")

    if select_cols:
        sel = select_cols
    sql = f"SELECT {', '.join(sel)} FROM [{TABLA}] WHERE 1=1 {extra_where}"
    return sql, []

# ---------- MODELOS ----------
class TopItem(BaseModel):
    nombre: str
    valor: float

class ResumenGrupo(BaseModel):
    grupo_inventario: str
    desde: str
    hasta: str
    total_unidades: float
    total_valor_subtotal: float
    ticket_promedio: float
    ventas_por_cliente: Dict[str, float]
    top_productos: List[TopItem]
    mensual_ventas: Dict[str, float]
    mensual_unidades: Dict[str, float]

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

class TopRespuesta(BaseModel):
    entidad: Literal["clientes","productos"]
    orden: Literal["mas","menos"]
    frecuencia: Literal["mensual","anual"]
    desde: str
    hasta: str
    grupo_inventario: Optional[str] = None
    categoria: Optional[str] = None
    top: List[Dict[str, object]]

# ---------- ROOT / HEALTH / METADATA ----------
@app.get("/")
def home():
    return {
        "ok": True,
        "service": "ventas-sqlite-api",
        "docs": "/docs",
        "health": "/health",
        "tabla": TABLA,
        "public": PUBLIC_MODE
    }

@app.get("/health")
def health():
    with get_conn() as conn:
        fec_col = COLS["Fecha"]
        dfc = pd.read_sql(f"SELECT count(*) as n FROM [{TABLA}] WHERE [{fec_col}] IS NOT NULL", conn)
        cnt = int(dfc.iloc[0]["n"])
    return {"ok": True, "public": PUBLIC_MODE, "db": str(DB_PATH), "tabla": TABLA, "cols": COLS, "rows_con_fecha": cnt}

@app.get("/tablas")
def tablas():
    return {"tablas": list_tables()}

@app.get("/schema")
def schema():
    with get_conn() as conn:
        df = pd.read_sql(f"PRAGMA table_info([{TABLA}])", conn)
    return {"tabla": TABLA, "columns": df.to_dict(orient="records")}

# ---------- RECARGA DESDE CSV (OneDrive) ----------
@app.post("/refresh_db", dependencies=[Depends(require_auth)])
def refresh_db():
    """
    Recarga la tabla principal TABLA desde el CSV configurado en CSV_URL.
    - Descarga el CSV de OneDrive.
    - Reemplaza completamente la tabla TABLA en ventas2025.sqlite.
    """
    if not CSV_URL:
        raise HTTPException(500, "CSV_URL no está definido en variables de entorno.")

    try:
        resp = requests.get(CSV_URL)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(500, f"No se pudo descargar el CSV: {e}")

    # Intentar leer el CSV (asumiendo delimitador ';' como en tus archivos)
    try:
        contenido = resp.content.decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(contenido), sep=";")
    except Exception as e:
        raise HTTPException(500, f"No se pudo leer el CSV: {e}")

    if df.empty:
        raise HTTPException(400, "El CSV no contiene filas.")

    # Reemplazar la tabla principal que la API está usando (TABLA)
    try:
        with get_conn() as conn:
            df.to_sql(TABLA, conn, if_exists="replace", index=False)
    except Exception as e:
        raise HTTPException(500, f"No se pudo actualizar la tabla {TABLA}: {e}")

    return {
        "ok": True,
        "mensaje": f"Tabla {TABLA} recargada desde CSV.",
        "filas": int(len(df))
    }

# ---------- consulta cliente (solo SUBTOTAL) ----------
@app.get("/consulta_cliente", response_model=ResumenCliente, dependencies=[Depends(require_auth)])
def consulta_cliente(
    cliente: Optional[str] = Query(None, min_length=2, description="Nombre o parte del nombre"),
    identificacion: Optional[str] = Query(None, description="Identificación exacta o parcial (solo números serán usados)"),
    desde: str   = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str   = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    grupo_inventario: Optional[str] = Query(None, description="Filtra por grupo de inventario (opcional)"),
    categoria: Optional[str] = Query(None, description="Filtra por categoría (opcional)")
):
    ensure_period(desde, hasta)
    val_col = COLS.get("Subtotal")
    if not val_col:
        raise HTTPException(500, "No existe columna Subtotal en la base. Corrige la estructura.")

    cli_col = COLS["Cliente"]; fec_col = COLS["Fecha"]
    grp_col = COLS.get("GrupoInventario")
    cat_col = COLS.get("Categoria")
    id_col  = COLS.get("Identificacion")

    where = ["date([" + fec_col + "]) BETWEEN ? AND ?"]
    params: List = [desde, hasta]

    filtro_det = []
    cliente_det = None
    cliente_id_det = None

    if identificacion:
        if not id_col:
            raise HTTPException(400, "No hay columna de Identificación en la base (ver /schema).")
        ident_digits = extract_digits(identificacion)
        if not ident_digits:
            raise HTTPException(422, "Identificación inválida (no se encontraron dígitos).")
        clean_sql = (
            f"lower(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace("
            f"replace([{id_col}], '-', ''), '.', ''), ' ', ''), '/', ''), '+', ''), ',', ''), '(', ''), ')', ''), '_', ''), '#', ''), '*', ''))"
        )
        where.append(f"{clean_sql} LIKE ?")
        params.append(f"%{ident_digits}%")
        filtro_det.append(f"Identificación~{ident_digits}")
    elif cliente:
        where.append(f"lower(REPLACE([{cli_col}], ' ', '')) LIKE ?")
        params.append(f"%{strip_accents_spaces_lower(cliente)}%")
        filtro_det.append(f"Cliente~{cliente}")
    else:
        raise HTTPException(422, "Debes enviar 'cliente' o 'identificacion'.")

    if grupo_inventario and grp_col:
        where.append(f"lower(REPLACE([{grp_col}], ' ', '')) = ?")
        params.append(strip_accents_spaces_lower(grupo_inventario))
        filtro_det.append(f"Grupo={grupo_inventario}")

    if categoria:
        if not cat_col:
            raise HTTPException(400, "No existe columna de Categoría en la base.")
        where.append(f"lower(REPLACE([{cat_col}], ' ', '')) = ?")
        params.append(strip_accents_spaces_lower(categoria))
        filtro_det.append(f"Categoria={categoria}")

    extra_where = " AND " + " AND ".join(where)
    base_sql, _ = build_base_select(val_col, extra_where=extra_where)

    try:
        with get_conn() as conn:
            df = pd.read_sql(base_sql, conn, params=params)
    except Exception as e:
        raise HTTPException(400, f"Consulta inválida ({' & '.join(filtro_det)}): {e}")

    if df.empty:
        raise HTTPException(404, f"Sin datos para ese filtro ({' & '.join(filtro_det)}) en {desde} → {hasta}.")

    # Normalización de números
    df["Cantidad"] = parse_number_series(df["Cantidad"])
    df["Subtotal"] = parse_number_series(df["Subtotal"])
    if df["Subtotal"].sum() == 0:
        raise HTTPException(404, "No hay registros con valores en el campo 'Subtotal' para este filtro.")

    # Resolver homónimos y set de cliente/ID
    if identificacion and id_col:
        with get_conn() as conn:
            df_id = pd.read_sql(
                f"SELECT [{id_col}] AS Ident, [{cli_col}] AS Cliente FROM [{TABLA}] "
                f"WHERE date([{fec_col}]) BETWEEN ? AND ?",
                conn, params=[desde, hasta]
            )
        if not df_id.empty:
            cliente_det = df["Cliente"].value_counts().idxmax()
            cliente_id_det = df_id[df_id["Cliente"] == cliente_det]["Ident"].value_counts().idxmax()
    else:
        cliente_det = df["Cliente"].value_counts().idxmax()

    if cliente_det is not None:
        df = df[df["Cliente"] == cliente_det].copy()

    total_unidades = float(df["Cantidad"].sum())
    total_valor    = float(df["Subtotal"].sum())
    ticket_prom    = float(total_valor / total_unidades) if total_unidades else 0.0

    por_grupo = (
        df.groupby("GrupoInventario", dropna=False)["Subtotal"]
          .sum().sort_values(ascending=False).round(2).to_dict()
    )

    top_prod_df = (
        df.groupby("Producto", dropna=False)["Subtotal"]
          .sum().sort_values(ascending=False).head(10).reset_index()
          .rename(columns={"Subtotal": "valor"})
    )
    top_list = [
        TopItem(
            nombre=("N/A" if pd.isna(r["Producto"]) else str(r["Producto"])),
            valor=float(r["valor"])
        )
        for _, r in top_prod_df.iterrows()
    ]

    df["YYYYMM"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.strftime("%Y-%m")
    mens_val = df.groupby("YYYYMM", dropna=False)["Subtotal"].sum().round(2).to_dict()
    mens_qty = df.groupby("YYYYMM", dropna=False)["Cantidad"].sum().round(2).to_dict()

    return ResumenCliente(
        cliente=str(cliente_det) if cliente_det is not None else "N/A",
        cliente_id=str(cliente_id_det) if cliente_id_det else None,
        desde=desde,
        hasta=hasta,
        total_unidades=round(total_unidades, 2),
        total_valor_subtotal=round(total_valor, 2),
        ticket_promedio=round(ticket_prom, 2),
        ventas_por_grupo={ (k if k is not None else "N/A"): float(v) for k,v in por_grupo.items() },
        top_productos=top_list,
        mensual_ventas={k: float(v) for k, v in mens_val.items()},
        mensual_unidades={k: float(v) for k, v in mens_qty.items()}
    )

# ---------- consulta por GRUPO DE INVENTARIO ----------
@app.get("/consulta_grupo", response_model=ResumenGrupo, dependencies=[Depends(require_auth)])
def consulta_grupo(
    grupo_inventario: str = Query(..., min_length=2, description="Nombre o parte del grupo de inventario"),
    desde: str   = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str   = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    categoria: Optional[str] = Query(None, description="Filtra por categoría (opcional)")
):
    """
    Resumen agregado por Grupo de inventario:
    - Total unidades y valor
    - Ticket promedio (valor / unidades)
    - Ventas por cliente
    - Top productos
    - Serie mensual (unidades y valor)
    """
    ensure_period(desde, hasta)
    val_col = COLS.get("Subtotal")
    if not val_col:
        raise HTTPException(500, "No existe columna Subtotal en la base. Corrige la estructura.")

    grp_col = COLS.get("GrupoInventario")
    if not grp_col:
        raise HTTPException(400, "No existe columna de GrupoInventario en la base. Revisa /schema.")
    cli_col = COLS["Cliente"]
    fec_col = COLS["Fecha"]
    cat_col = COLS.get("Categoria")
    pro_col = COLS["Producto"]
    qty_col = COLS["Cantidad"]

    where = [f"date([{fec_col}]) BETWEEN ? AND ?"]
    params: List = [desde, hasta]

    # Filtro robusto por grupo (normalizado)
    norm_val = py_norm_text(grupo_inventario)
    norm_col = sql_norm_column(grp_col)
    where.append(f"{norm_col} LIKE ?")
    params.append(f"%{norm_val}%")

    # Filtro opcional por categoría
    if categoria and cat_col:
        cat_norm = py_norm_text(categoria)
        cat_norm_col = sql_norm_column(cat_col)
        where.append(f"{cat_norm_col} LIKE ?")
        params.append(f"%{cat_norm}%")

    extra_where = " AND " + " AND ".join(where)

    # Usamos la misma función base para traer:
    # Cliente, Fecha, Producto, Cantidad, Subtotal, GrupoInventario, Categoria
    base_sql, _ = build_base_select(val_col, extra_where=extra_where)

    try:
        with get_conn() as conn:
            df = pd.read_sql(base_sql, conn, params=params)
    except Exception as e:
        raise HTTPException(400, f"Consulta inválida para grupo '{grupo_inventario}': {e}")

    if df.empty:
        raise HTTPException(404, f"No hay ventas para el grupo '{grupo_inventario}' en {desde} → {hasta}.")

    # Normalizar números
    df["Cantidad"] = parse_number_series(df["Cantidad"])
    df["Subtotal"] = parse_number_series(df["Subtotal"])
    if df["Subtotal"].sum() == 0:
        raise HTTPException(404, "No hay registros con valores en 'Subtotal' para este grupo.")

    # Determinar el grupo REAL dominante (por si el filtro fue parcial)
    if "GrupoInventario" in df.columns:
        grupo_real = df["GrupoInventario"].value_counts().idxmax()
        df = df[df["GrupoInventario"] == grupo_real].copy()
    else:
        grupo_real = grupo_inventario

    # Totales
    total_unidades = float(df["Cantidad"].sum())
    total_valor    = float(df["Subtotal"].sum())
    ticket_prom    = float(total_valor / total_unidades) if total_unidades else 0.0

    # Ventas por cliente
    ventas_cli = (
        df.groupby("Cliente", dropna=False)["Subtotal"]
          .sum().sort_values(ascending=False).round(2).to_dict()
    )

    # Top productos
    top_prod_df = (
        df.groupby("Producto", dropna=False)["Subtotal"]
          .sum().sort_values(ascending=False).head(10).reset_index()
          .rename(columns={"Subtotal": "valor"})
    )
    top_list = [
        TopItem(
            nombre=("N/A" if pd.isna(r["Producto"]) else str(r["Producto"])),
            valor=float(r["valor"])
        )
        for _, r in top_prod_df.iterrows()
    ]

    # Series mensuales
    df["YYYYMM"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.strftime("%Y-%m")
    mens_val = df.groupby("YYYYMM", dropna=False)["Subtotal"].sum().round(2).to_dict()
    mens_qty = df.groupby("YYYYMM", dropna=False)["Cantidad"].sum().round(2).to_dict()

    return ResumenGrupo(
        grupo_inventario=str(grupo_real),
        desde=desde,
        hasta=hasta,
        total_unidades=round(total_unidades, 2),
        total_valor_subtotal=round(total_valor, 2),
        ticket_promedio=round(ticket_prom, 2),
        ventas_por_cliente={ (k if k is not None else "N/A"): float(v) for k,v in ventas_cli.items() },
        top_productos=top_list,
        mensual_ventas={k: float(v) for k, v in mens_val.items()},
        mensual_unidades={k: float(v) for k, v in mens_qty.items()}
    )

# ---------- informe (texto) ----------
@app.get("/informe_cliente", dependencies=[Depends(require_auth)])
def informe_cliente(
    cliente: Optional[str] = None,
    identificacion: Optional[str] = None,
    desde: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    grupo_inventario: Optional[str] = None,
    categoria: Optional[str] = None
):
    try:
        r = consulta_cliente(
            cliente=cliente,
            identificacion=identificacion,
            desde=desde,
            hasta=hasta,
            grupo_inventario=grupo_inventario,
            categoria=categoria
        )
    except HTTPException as e:
        if e.status_code == 404:
            target = identificacion or cliente or "filtro"
            return {"informe": f"No encontré ventas para '{target}' en {desde} → {hasta}."}
        raise

    lineas = [ "🧾 Informe de cliente" ]
    if r.cliente_id:
        lineas.append(f"Cliente: {r.cliente} (ID: {r.cliente_id})")
    else:
        lineas.append(f"Cliente: {r.cliente}")
    if grupo_inventario:
        lineas.append(f"Grupo inventario: {grupo_inventario}")
    if categoria:
        lineas.append(f"Categoría: {categoria}")
    lineas += [
        f"Periodo: {r.desde} → {r.hasta}",
        f"Unidades: {r.total_unidades:,.0f}",
        f"Ventas (Subtotal): ${r.total_valor_subtotal:,.2f}",
        f"Ticket promedio: ${r.ticket_promedio:,.2f}",
        "",
        "Ventas por grupo (Top):"
    ]
    port = sorted(r.ventas_por_grupo.items(), key=lambda x: x[1], reverse=True)
    for k, v in port[:10]:
        lineas.append(f"  - {k}: ${v:,.2f}")

    lineas.append("")
    lineas.append("Top productos (Subtotal):")
    for i, tp in enumerate(r.top_productos[:10], 1):
        lineas.append(f"  {i}. {tp.nombre} — ${tp.valor:,.2f}")

    lineas.append("")
    lineas.append("Mensual (Subtotal):")
    for m, v in sorted(r.mensual_ventas.items()):
        lineas.append(f"  {m}: ${v:,.2f}")

    return {"informe": "\n".join(lineas)}

# ---------- TOPS globales ----------
@app.get("/tops", response_model=TopRespuesta, dependencies=[Depends(require_auth)])
def tops(
    entidad: Literal["clientes","productos"] = Query(..., description="Entidad objetivo del top"),
    orden: Literal["mas","menos"] = Query("mas", description="‘mas’ o ‘menos’"),
    frecuencia: Literal["mensual","anual"] = Query("anual"),
    desde: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    grupo_inventario: Optional[str] = Query(None, description="Filtro opcional por grupo de inventario"),
    categoria: Optional[str] = Query(None, description="Filtro opcional por categoría"),
    limite: int = Query(10, ge=1, le=100)
):
    ensure_period(desde, hasta)
    val_col = COLS.get("Subtotal")
    if not val_col:
        raise HTTPException(500, "No existe columna Subtotal en la base.")

    cli_col = COLS["Cliente"]; fec_col = COLS["Fecha"]
    grp_col = COLS.get("GrupoInventario")
    cat_col = COLS.get("Categoria")
    pro_col = COLS["Producto"]

    where = [f"date([{fec_col}]) BETWEEN ? AND ?"]
    params: List = [desde, hasta]

    # Filtros robustos por grupo/categoría
    add_text_filter(where, params, grupo_inventario, grp_col)
    add_text_filter(where, params, categoria,        cat_col)

    target_col = f"[{cli_col}]" if entidad == "clientes" else f"[{pro_col}]"

    base_sql = f"""
        SELECT {target_col} AS Nombre,
               [{val_col}]  AS Subtotal,
               [{fec_col}]  AS Fecha
               {"," + f"[{grp_col}] AS GrupoInventario" if grp_col else ""}
               {"," + f"[{cat_col}] AS Categoria" if cat_col else ""}
        FROM [{TABLA}]
        WHERE {" AND ".join(where)}
    """

    with get_conn() as conn:
        df = pd.read_sql(base_sql, conn, params=params)

    if df.empty:
        raise HTTPException(404, "No hay registros para el periodo/filtro indicado.")

    df["Subtotal"] = parse_number_series(df["Subtotal"])

    if df["Subtotal"].sum() == 0:
        raise HTTPException(404, "No hay valores en 'Subtotal' para el periodo/filtro indicado.")

    df["Nombre"] = df["Nombre"].fillna("N/A")

    if frecuencia == "mensual":
        df["Periodo"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.strftime("%Y-%m")
        g = df.groupby(["Periodo","Nombre"], dropna=False)["Subtotal"].sum().reset_index()
        resultado: List[Dict[str, object]] = []
        for periodo, chunk in g.groupby("Periodo"):
            chunk = chunk.sort_values("Subtotal", ascending=(orden == "menos"))
            head = chunk.head(limite)
            for _, r in head.iterrows():
                resultado.append({"periodo": periodo, "nombre": str(r["Nombre"]), "valor": float(round(r["Subtotal"],2))})
    else:
        g = df.groupby("Nombre", dropna=False)["Subtotal"].sum().reset_index()
        g = g.sort_values("Subtotal", ascending=(orden == "menos")).head(limite)
        resultado = [{"nombre": str(r["Nombre"]), "valor": float(round(r["Subtotal"],2))} for _, r in g.iterrows()]

    return TopRespuesta(
        entidad=entidad, orden=orden, frecuencia=frecuencia,
        desde=desde, hasta=hasta,
        grupo_inventario=grupo_inventario, categoria=categoria,
        top=resultado
    )

# ---------- Series (MENSUAL / ANUAL) LIMPIAS ----------
def _ensure_period_or_default(desde: Optional[str], hasta: Optional[str]) -> Tuple[str,str]:
    """
    Si no envían desde/hasta, usa todo el rango disponible en la base (por fecha).
    """
    fec_col = COLS["Fecha"]
    with get_conn() as conn:
        r = pd.read_sql(f"SELECT MIN(date([{fec_col}])) AS d1, MAX(date([{fec_col}])) AS d2 FROM [{TABLA}]", conn).iloc[0]
    d1 = str(r["d1"]) if pd.notna(r["d1"]) else "1900-01-01"
    d2 = str(r["d2"]) if pd.notna(r["d2"]) else "2100-12-31"
    return (desde or d1, hasta or d2)

def _build_sales_df(desde: str, hasta: str,
                    grupo_inventario: Optional[str],
                    categoria: Optional[str],
                    producto: Optional[str]) -> pd.DataFrame:
    """
    Devuelve un DataFrame con: Fecha, Subtotal, GrupoInventario, Categoria, Producto
    y aplica filtros robustos si se piden.
    """
    val_col = COLS.get("Subtotal")
    if not val_col:
        raise HTTPException(500, "No existe columna Subtotal en la base.")
    fec_col = COLS["Fecha"]
    grp_col = COLS.get("GrupoInventario")
    cat_col = COLS.get("Categoria")
    pro_col = COLS["Producto"]

    where = [f"date([{fec_col}]) BETWEEN ? AND ?"]
    params: List[str] = [desde, hasta]

    # --- normalizamos COLUMNA y VALOR de forma consistente ---
    def add_like(value: Optional[str], col: Optional[str]):
        if value and col:
            norm_col = sql_norm_column(col)
            norm_val = py_norm_text(value)
            where.append(f"{norm_col} LIKE ?")
            params.append(f"%{norm_val}%")

    add_like(grupo_inventario, grp_col)
    add_like(categoria,        cat_col)
    add_like(producto,         pro_col)

    sql = f"""
      SELECT [{fec_col}] AS Fecha,
             [{val_col}] AS Subtotal,
             { (f"[{grp_col}] AS GrupoInventario" if grp_col else "'N/A' AS GrupoInventario") },
             { (f"[{cat_col}] AS Categoria"      if cat_col else "'N/A' AS Categoria") },
             [{pro_col}]  AS Producto
      FROM [{TABLA}]
      WHERE {" AND ".join(where)}
    """
    try:
        with get_conn() as conn:
            df = pd.read_sql(sql, conn, params=params)
    except Exception as e:
        raise HTTPException(400, f"Consulta inválida: {e}")

    if df.empty:
        raise HTTPException(404, "No hay registros para el filtro/periodo indicado.")

    df["Subtotal"] = parse_number_series(df["Subtotal"])
    if df["Subtotal"].sum() == 0:
        raise HTTPException(404, "No hay valores en 'Subtotal' para el filtro/periodo indicado.")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])
    return df

@app.get("/ventas_mensuales_vista", dependencies=[Depends(require_auth)])
def ventas_mensuales_vista(
    desde: Optional[str] = Query(None),
    hasta: Optional[str] = Query(None),
    grupo_inventario: Optional[str] = Query(None),
    categoria: Optional[str] = Query(None),
    producto: Optional[str] = Query(None)
):
    d, h = _ensure_period_or_default(desde, hasta)
    ensure_period(d, h)

    df = _build_sales_df(d, h, grupo_inventario, categoria, producto)
    df["Mes"] = df["Fecha"].dt.strftime("%Y-%m")
    mens = df.groupby("Mes", dropna=False)["Subtotal"].sum().sort_index()

    return {
        "ok": True,
        "filtros": {"desde": d, "hasta": h,
                    "grupo_inventario": grupo_inventario,
                    "categoria": categoria,
                    "producto": producto},
        "ventas_mensuales": [{"Mes": k, "Total": float(round(v,2))} for k, v in mens.items()]
    }

@app.get("/ventas_anuales_vista", dependencies=[Depends(require_auth)])
def ventas_anuales_vista(
    desde: Optional[str] = Query(None),
    hasta: Optional[str] = Query(None),
    grupo_inventario: Optional[str] = Query(None),
    categoria:        Optional[str] = Query(None),
    producto:         Optional[str] = Query(None)
):
    d, h = _ensure_period_or_default(desde, hasta)
    ensure_period(d, h)

    df = _build_sales_df(d, h, grupo_inventario, categoria, producto)
    df["Anio"] = df["Fecha"].dt.year.astype(str)
    anual = df.groupby("Anio", dropna=False)["Subtotal"].sum().sort_index()

    return {
        "ok": True,
        "filtros": {"desde": d, "hasta": h,
                    "grupo_inventario": grupo_inventario,
                    "categoria": categoria,
                    "producto": producto},
        "ventas_anuales": [{"Anio": k, "Total": float(round(v,2))} for k, v in anual.items()]
    }

# --------- ALIAS explícitos (útiles para Actions / agentes) ----------
@app.get("/ventas_mensuales_por_grupo", dependencies=[Depends(require_auth)])
def ventas_mensuales_por_grupo(
    # acepta ambos nombres
    grupo_inventario: Optional[str] = Query(None, alias="grupo_inventario"),
    grupo: Optional[str] = Query(None, alias="grupo"),
    desde: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    gi = grupo_inventario or grupo
    if not gi:
        raise HTTPException(422, "Debe enviar 'grupo_inventario' o 'grupo'.")
    return ventas_mensuales_vista(desde=desde, hasta=hasta,
                                  grupo_inventario=gi,
                                  categoria=None, producto=None)

@app.get("/ventas_mensuales_por_categoria", dependencies=[Depends(require_auth)])
def ventas_mensuales_por_categoria(
    categoria: Optional[str] = Query(None, alias="categoria"),
    category: Optional[str] = Query(None, alias="category"),
    desde: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    cat = categoria or category
    if not cat:
        raise HTTPException(422, "Debe enviar 'categoria' (o 'category').")
    return ventas_mensuales_vista(desde=desde, hasta=hasta,
                                  grupo_inventario=None,
                                  categoria=cat, producto=None)

@app.get("/ventas_mensuales_por_producto", dependencies=[Depends(require_auth)])
def ventas_mensuales_por_producto(
    producto: Optional[str] = Query(None, alias="producto"),
    product: Optional[str]  = Query(None, alias="product"),
    desde: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    prod = producto or product
    if not prod:
        raise HTTPException(422, "Debe enviar 'producto' (o 'product').")
    return ventas_mensuales_vista(desde=desde, hasta=hasta,
                                  grupo_inventario=None,
                                  categoria=None, producto=prod)

@app.get("/ventas_anuales_por_grupo", dependencies=[Depends(require_auth)])
def ventas_anuales_por_grupo(
    grupo_inventario: Optional[str] = Query(None, alias="grupo_inventario"),
    grupo: Optional[str] = Query(None, alias="grupo"),
    desde: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    gi = grupo_inventario or grupo
    if not gi:
        raise HTTPException(422, "Debe enviar 'grupo_inventario' o 'grupo'.")
    return ventas_anuales_vista(desde=desde, hasta=hasta,
                                grupo_inventario=gi,
                                categoria=None, producto=None)

@app.get("/ventas_anuales_por_categoria", dependencies=[Depends(require_auth)])
def ventas_anuales_por_categoria(
    categoria: Optional[str] = Query(None, alias="categoria"),
    category: Optional[str] = Query(None, alias="category"),
    desde: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    cat = categoria or category
    if not cat:
        raise HTTPException(422, "Debe enviar 'categoria' (o 'category').")
    return ventas_anuales_vista(desde=desde, hasta=hasta,
                                grupo_inventario=None,
                                categoria=cat, producto=None)

@app.get("/ventas_anuales_por_producto", dependencies=[Depends(require_auth)])
def ventas_anuales_por_producto(
    producto: Optional[str] = Query(None, alias="producto"),
    product: Optional[str]  = Query(None, alias="product"),
    desde: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    prod = producto or product
    if not prod:
        raise HTTPException(422, "Debe enviar 'producto' (o 'product').")
    return ventas_anuales_vista(desde=desde, hasta=hasta,
                                grupo_inventario=None,
                                categoria=None, producto=prod)

# ---------- Descubrimiento de valores (para evitar "nombre exacto") ----------
def _listar_unicos(col_real: str, contiene: Optional[str], limite: int = 200):
    try:
        with get_conn() as conn:
            where = f"WHERE [{col_real}] IS NOT NULL"
            params: List[str] = []
            if contiene:
                norm_col = sql_norm_column(col_real)
                norm_val = py_norm_text(contiene)
                where += f" AND {norm_col} LIKE ?"
                params.append(f"%{norm_val}%")
            sql = f"""
                SELECT [{col_real}] AS valor, COUNT(*) AS conteo
                FROM [{TABLA}]
                {where}
                GROUP BY [{col_real}]
                ORDER BY conteo DESC
                LIMIT ?
            """
            params.append(int(limite))
            df = pd.read_sql(sql, conn, params=params)
        return {"items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(400, f"Consulta inválida en valores/{col_real}: {e}")

@app.get("/valores/grupos", dependencies=[Depends(require_auth)])
def valores_grupos(contiene: Optional[str] = None, limite: int = 200):
    col = COLS.get("GrupoInventario")
    if not col:
        raise HTTPException(400, "No existe columna de GrupoInventario en la base.")
    return _listar_unicos(col, contiene, limite)

@app.get("/valores/categorias", dependencies=[Depends(require_auth)])
def valores_categorias(contiene: Optional[str] = None, limite: int = 200):
    col = COLS.get("Categoria")
    if not col:
        raise HTTPException(400, "No existe columna de Categoría en la base.")
    return _listar_unicos(col, contiene, limite)

@app.get("/valores/productos", dependencies=[Depends(require_auth)])
def valores_productos(contiene: Optional[str] = None, limite: int = 200):
    col = COLS.get("Producto")
    if not col:
        raise HTTPException(400, "No existe columna de Producto en la base.")
    return _listar_unicos(col, contiene, limite)
