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
app = FastAPI(title="Ventas API (SQLite)", version="2.4.1")

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
    # Evitar bloqueos de thread con Uvicorn
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

    # Identificación (ampliamos alias comunes)
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
                            pick_fuzzy(norm_map, "grupoinventario", "grupo", "linea", "categoria"))

    # Portafolio (compat)
    m["Portafolio"] = (norm_map.get(normalize("Portafolio")) or m["GrupoInventario"])

    # Producto
    m["Producto"] = (norm_map.get(normalize("Nombre producto")) or
                     pick_fuzzy(norm_map, "nombreproducto", "producto", "codigo", "codigoproducto"))

    # Cantidad
    m["Cantidad"] = (norm_map.get(normalize("Cantidad vendida")) or
                     pick_fuzzy(norm_map, "cantidadvendida", "cantidad", "unidades", "unid", "cant", "qty"))

    # Valores (SOLO Subtotal)
    m["Subtotal"] = norm_map.get(normalize("Subtotal"))

    missing = [k for k in ["Cliente","Fecha","Cantidad","Subtotal"] if not m.get(k)]
    if missing:
        raise HTTPException(400, f"Faltan columnas mínimas en [{tbl}]: {missing}. Revisa nombres.")
    return m

# ---------- modelos ----------
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
    mensual_ventas: Dict[str, float]   # YYYY-MM -> subtotal
    mensual_unidades: Dict[str, float] # YYYY-MM -> qty

class TopRespuesta(BaseModel):
    entidad: Literal["clientes","productos"]
    orden: Literal["mas","menos"]
    frecuencia: Literal["mensual","anual"]
    desde: str
    hasta: str
    grupo_inventario: Optional[str] = None
    top: List[Dict[str, object]]     # [{periodo?, nombre, valor}]

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

def sql_number(col: str) -> str:
    """
    Convierte texto monetario/numérico a float en SQLite sin usar funciones no soportadas.
    Maneja:
      - LATAM: 1.234.567,89
      - EN:    1,234,567.89
    Limpia símbolos: $, espacios (incluido NBSP), #, *, (), saltos.
    """
    # NBSP literal ' ' (u00A0) incluido directamente en el string
    # NO quitamos ',' ni '.' en 'clean' para poder decidir el formato después.
    clean = (
        f"replace(replace(replace(replace(replace(replace(replace(replace(replace([{col}],"
        f"'$',''),' ',''),' ','') , '#',''), '*',''), '(' ,''), ')',''), char(10), ''), char(9), '')"
    )
    # Si tiene coma y punto -> LATAM: quitar puntos (miles) y cambiar coma por punto
    latam = f"cast(replace(replace({clean}, '.', ''), ',', '.') as float)"
    # Si tiene solo coma -> usar coma como decimal
    only_comma = f"cast(replace({clean}, ',', '.') as float)"
    # En otro caso -> EN: quitar comas (miles)
    enfmt = f"cast(replace({clean}, ',', '') as float)"
    return (
        f"(case when instr({clean}, ',')>0 and instr({clean}, '.')>0 then {latam} "
        f" when instr({clean}, ',')>0 and instr({clean}, '.')=0 then {only_comma} "
        f" else {enfmt} end)"
    )

# ---------- helpers SQL ----------
def build_base_select(val_col: str, extra_where: str = "", select_cols: Optional[List[str]] = None) -> Tuple[str, List]:
    cli_col = COLS["Cliente"]; fec_col = COLS["Fecha"]; qty_col = COLS["Cantidad"]
    pro_col = COLS["Producto"]; grp_col = COLS["GrupoInventario"]

    qty_expr = sql_number(qty_col)
    val_expr = sql_number(val_col)

    sel = [
        f"[{cli_col}]  AS Cliente",
        f"[{fec_col}]  AS Fecha",
        f"[{pro_col}]  AS Producto",
        f"{qty_expr}  AS Cantidad",
        f"{val_expr}  AS Subtotal",
    ]
    if grp_col:
        sel.insert(2, f"[{grp_col}] AS GrupoInventario")
    else:
        sel.insert(2, "'N/A' AS GrupoInventario")

    if select_cols:
        sel = select_cols
    sql = f"SELECT {', '.join(sel)} FROM [{TABLA}] WHERE 1=1 {extra_where}"
    return sql, []

# ---------- endpoints básicos ----------
@app.get("/health")
def health():
    return {"ok": True, "public": PUBLIC_MODE, "db": str(DB_PATH), "tabla": TABLA, "cols": COLS}

@app.get("/debug_ping")
def debug_ping():
    # chequeo rápido de tabla/fecha
    fec_col = COLS["Fecha"]
    sql = f'SELECT count(*) as n FROM [{TABLA}] WHERE [{fec_col}] IS NOT NULL'
    with get_conn() as conn:
        try:
            df = pd.read_sql(sql, conn)
            return {"sql": sql, "count": int(df.iloc[0]["n"])}
        except Exception as e:
            raise HTTPException(500, f"DEBUG ping error: {e}")

@app.get("/tablas")
def tablas():
    return {"tablas": list_tables()}

@app.get("/schema")
def schema():
    with get_conn() as conn:
        df = pd.read_sql(f"PRAGMA table_info([{TABLA}])", conn)
    return {"tabla": TABLA, "columns": df.to_dict(orient="records")}

# ---------- consulta cliente (solo SUBTOTAL) ----------
class TopItem(BaseModel):
    nombre: str
    valor: float

@app.get("/consulta_cliente", response_model=ResumenCliente, dependencies=[Depends(require_auth)])
def consulta_cliente(
    cliente: Optional[str] = Query(None, min_length=2, description="Nombre o parte del nombre"),
    identificacion: Optional[str] = Query(None, description="Identificación exacta o parcial (solo números serán usados)"),
    desde: str   = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str   = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    grupo_inventario: Optional[str] = Query(None, description="Filtra por grupo de inventario (opcional)")
):
    ensure_period(desde, hasta)
    val_col = COLS.get("Subtotal")
    if not val_col:
        raise HTTPException(500, "No existe columna Subtotal en la base. Corrige la estructura.")

    cli_col = COLS["Cliente"]; fec_col = COLS["Fecha"]; grp_col = COLS["GrupoInventario"]
    id_col  = COLS.get("Identificacion")

    where = ["date([" + fec_col + "]) BETWEEN ? AND ?"]
    params: List = [desde, hasta]

    filtro_det = ""
    cliente_det = None
    cliente_id_det = None

    clean_sql = None
    ident_digits = None

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
        filtro_det = f"Identificación ~ {ident_digits}"

    elif cliente:
        where.append(f"lower(REPLACE([{cli_col}], ' ', '')) LIKE ?")
        params.append(f"%{strip_accents_spaces_lower(cliente)}%")
        filtro_det = f"Cliente ~ {cliente}"
    else:
        raise HTTPException(422, "Debes enviar 'cliente' o 'identificacion'.")

    if grupo_inventario and grp_col:
        where.append(f"lower([{grp_col}]) = lower(?)")
        params.append(grupo_inventario)

    # Validación existencia por identificación (independiente de Subtotal)
    if identificacion:
        with get_conn() as conn:
            exist_sql = f"""
                SELECT 1
                FROM [{TABLA}]
                WHERE date([{fec_col}]) BETWEEN ? AND ?
                  AND {clean_sql} LIKE ?
                LIMIT 1
            """
            exists = pd.read_sql(exist_sql, conn, params=[desde, hasta, f"%{ident_digits}%"])
            if exists.empty:
                raise HTTPException(404, f"No se encontró la identificación enviada en el período: {ident_digits}.")

    # Consulta principal
    extra_where = " AND " + " AND ".join(where)
    base_sql, _ = build_base_select(val_col, extra_where=extra_where)

    with get_conn() as conn:
        try:
            df = pd.read_sql(base_sql, conn, params=params)
        except Exception as e:
            raise HTTPException(500, f"Error SQL en consulta_cliente: {e}")

    if df.empty:
        raise HTTPException(404, f"Sin datos para ese filtro ({filtro_det}) en {desde} → {hasta}.")

    # Validación de Subtotal > 0
    if df["Subtotal"].fillna(0).sum() == 0:
        if identificacion:
            raise HTTPException(404, f"La identificación coincide, pero 'Subtotal' es 0/vacío en el período {desde} → {hasta}.")
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

    # Seguridad extra (ya vienen normalizados)
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0.0)
    df["Subtotal"] = pd.to_numeric(df["Subtotal"], errors="coerce").fillna(0.0)

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

# ---------- informe (texto) ----------
@app.get("/informe_cliente", dependencies=[Depends(require_auth)])
def informe_cliente(
    cliente: Optional[str] = None,
    identificacion: Optional[str] = None,
    desde: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    hasta: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    grupo_inventario: Optional[str] = None
):
    try:
        r = consulta_cliente(cliente=cliente, identificacion=identificacion, desde=desde, hasta=hasta, grupo_inventario=grupo_inventario)
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
    limite: int = Query(10, ge=1, le=100)
):
    ensure_period(desde, hasta)
    val_col = COLS.get("Subtotal")
    if not val_col:
        raise HTTPException(500, "No existe columna Subtotal en la base.")

    cli_col = COLS["Cliente"]; fec_col = COLS["Fecha"]; grp_col = COLS["GrupoInventario"]
    pro_col = COLS["Producto"]

    where = [f"date([{fec_col}]) BETWEEN ? AND ?"]
    params: List = [desde, hasta]
    if grupo_inventario and grp_col:
        where.append(f"lower([{grp_col}]) = lower(?)")
        params.append(grupo_inventario)

    target_col = f"[{cli_col}]" if entidad == "clientes" else f"[{pro_col}]"

    base_sel = [
        f"{target_col} AS Nombre",
        f"{sql_number(val_col)} AS Subtotal",
        f"[{fec_col}] AS Fecha"
    ]
    if grp_col:
        base_sel.append(f"[{grp_col}] AS GrupoInventario")

    base_sql = f"""
        SELECT {", ".join(base_sel)}
        FROM [{TABLA}]
        WHERE {" AND ".join(where)}
    """

    with get_conn() as conn:
        try:
            df = pd.read_sql(base_sql, conn, params=params)
        except Exception as e:
            raise HTTPException(500, f"Error SQL en tops: {e}")

    if df.empty:
        raise HTTPException(404, "No hay registros para el periodo/filtro indicado.")
    df["Subtotal"] = pd.to_numeric(df["Subtotal"], errors="coerce").fillna(0.0)
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
        desde=desde, hasta=hasta, grupo_inventario=grupo_inventario,
        top=resultado
    )
