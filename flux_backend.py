import pandas as pd
import numpy as np
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st

# Configuración para evitar errores de hilos con Matplotlib
matplotlib.use('Agg')

# ============================================================
# 1. Helpers de Formato y Fechas
# ============================================================
def parse_amount_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)

    x = s.astype(str).str.strip()
    x = x.str.replace(r"[^\d\-,\.]", "", regex=True)

    def _to_float(v: str):
        if v is None: return np.nan
        v = v.strip()
        if v in ("", "-", ".", ","): return np.nan
        has_dot = "." in v
        has_comma = "," in v
        if has_dot and has_comma:
            if v.rfind(",") > v.rfind("."): v2 = v.replace(".", "").replace(",", ".")
            else: v2 = v.replace(",", "")
            return pd.to_numeric(v2, errors="coerce")
        if has_comma and not has_dot:
            parts = v.split(",")
            if len(parts) == 2 and 1 <= len(parts[1]) <= 2: v2 = v.replace(",", ".")
            else: v2 = v.replace(",", "")
            return pd.to_numeric(v2, errors="coerce")
        if has_dot and not has_comma:
            parts = v.split(".")
            if len(parts) == 2 and 1 <= len(parts[1]) <= 2: v2 = v
            else: v2 = v.replace(".", "")
            return pd.to_numeric(v2, errors="coerce")
        return pd.to_numeric(v, errors="coerce")

    return x.map(_to_float).astype(float)

def months_diff(period_a: pd.Period, period_b: pd.Period) -> int:
    return (period_a.year - period_b.year) * 12 + (period_a.month - period_b.month)

def fmt_int(x): return "N/A" if pd.isna(x) else f"{x:,.0f}"
def fmt_money(x): return "N/A" if pd.isna(x) else f"${x:,.0f}"

# ============================================================
# 2. Carga y Limpieza (NUEVAS FUNCIONES)
# ============================================================
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Intenta leer el CSV probando separadores comunes (',' y ';')
    y codificaciones (utf-8, latin1).
    """
    try:
        # Intento 1: Separador estándar (,) y utf-8
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes), sep=",", low_memory=False)
        if len(df.columns) < 2:
            raise ValueError("Posible error de separador")
        return df
    except:
        try:
            # Intento 2: Separador punto y coma (;)
            df = pd.read_csv(pd.io.common.BytesIO(file_bytes), sep=";", low_memory=False)
            return df
        except:
            # Intento 3: Codificación latin1 (común en Excel viejos)
            return pd.read_csv(pd.io.common.BytesIO(file_bytes), sep=None, engine='python', encoding='latin1')

def clean_input_data(df: pd.DataFrame, col_fecha: str, col_id: str, col_monto: str) -> pd.DataFrame:
    """
    Deja solo las 3 columnas necesarias y limpia los datos sucios
    (IDs con nombres, fechas raras, montos con puntos/comas).
    """
    # 1. Seleccionar solo lo necesario y renombrar para estandarizar
    clean = df[[col_fecha, col_id, col_monto]].copy()
    clean.columns = ["fecha", "id", "monto"]

    # 2. LIMPIEZA DE ID (Crítica para "ID Nombre")
    clean["id"] = clean["id"].astype(str).str.strip()
    # Cortar en el primer espacio (ej: "12345 Juan" -> "12345")
    clean["id"] = clean["id"].str.split(pat=" ", n=1).str[0]
    # Eliminar caracteres raros (dejar solo letras, números, guiones y puntos)
    clean["id"] = clean["id"].str.replace(r"[^a-zA-Z0-9\-\.]", "", regex=True)

    # 3. LIMPIEZA DE MONTO
    clean["monto"] = parse_amount_series(clean["monto"])

    # 4. LIMPIEZA DE FECHA (dayfirst ayuda con formato Latam DD/MM/YYYY)
    clean["fecha"] = pd.to_datetime(clean["fecha"], dayfirst=True, errors="coerce")

    # 5. ELIMINAR FILAS BASURA (Sin fecha o sin monto)
    clean = clean.dropna(subset=["fecha", "monto"])
    
    return clean

# ============================================================
# 3. Cálculo de Cohortes (CON FRECUENCIA)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_historico(df: pd.DataFrame, col_fecha: str, col_id: str, col_monto: str,
                      outlier_pct: float, months_to_exclude: int):
    # Nota: df ya viene limpio de clean_input_data, pero renombramos por seguridad
    # para mantener compatibilidad con los nombres de argumentos.
    raw = df.rename(columns={col_fecha: "fecha", col_id: "id", col_monto: "monto"}).copy()
    
    # Aseguramos tipos (redundante si viene de clean_input_data, pero seguro)
    raw["fecha"] = pd.to_datetime(raw["fecha"], errors="coerce")
    raw["id"] = raw["id"].astype(str)
    raw["monto"] = parse_amount_series(raw["monto"])

    raw = raw.dropna(subset=["fecha", "id", "monto"]).copy()
    raw = raw[raw["monto"] > 0].copy()

    # --- AGREGACIÓN POR DÍA (Para Frecuencia Correcta) ---
    # Fusionamos compras del mismo día en una sola transacción
    raw = (raw.groupby(["id", "fecha"], as_index=False)
              .agg(monto=("monto", "sum")))

    raw["periodo_tx"] = raw["fecha"].dt.to_period("M")
    fecha_max = raw["fecha"].max()
    periodo_max = fecha_max.to_period("M")

    # Cohorte
    first_pos = raw.groupby("id")["fecha"].min()
    cohort_df = first_pos.to_frame(name="fecha_cohorte")
    cohort_df["periodo_cohorte"] = cohort_df["fecha_cohorte"].dt.to_period("M")
    cohort_df["edad_actual_meses"] = cohort_df["periodo_cohorte"].apply(lambda p: months_diff(periodo_max, p))

    raw = raw[raw["id"].isin(cohort_df.index)].copy()

    # Outliers
    tx_pos = raw["monto"]
    limite_outlier = float(tx_pos.quantile(1.0 - outlier_pct)) if len(tx_pos) else np.nan
    raw["es_outlier_tx"] = raw["monto"] >= limite_outlier if pd.notna(limite_outlier) else False
    outlier_customers = set(raw.loc[raw["es_outlier_tx"], "id"].unique())

    # Cliente-Mes
    raw = raw.join(cohort_df[["periodo_cohorte"]], on="id")
    cm = (raw.groupby(["id", "periodo_tx", "periodo_cohorte"], as_index=False)
            .agg(
                monto_neto_cliente_mes=("monto", "sum"),
                tx_count=("monto", "count"), 
                has_outlier_mes=("es_outlier_tx", "any")
            ))

    cm["mes_vida"] = cm.apply(lambda r: months_diff(r["periodo_tx"], r["periodo_cohorte"]), axis=1)
    cm = cm[cm["mes_vida"] >= 0].copy()
    cm["activo_mes"] = cm["monto_neto_cliente_mes"] > 0
    cm_arpu = cm[~cm["has_outlier_mes"]].copy()

    # Segmentos (Estricto 0-11 y 12-36)
    ids_nuevos = cohort_df[cohort_df["edad_actual_meses"].between(0, 11)].index
    ids_recurrentes = cohort_df[cohort_df["edad_actual_meses"].between(12, 36)].index

    # Frecuencia
    mask_new = (cm["id"].isin(ids_nuevos)) & (cm["mes_vida"].between(0, 11)) & (cm["activo_mes"])
    freq_new = cm.loc[mask_new, "tx_count"].mean() if mask_new.any() else 0.0

    mask_rec = (cm["id"].isin(ids_recurrentes)) & (cm["mes_vida"].between(12, 36)) & (cm["activo_mes"])
    freq_rec = cm.loc[mask_rec, "tx_count"].mean() if mask_rec.any() else 0.0

    # KPIs Globales
    win = pd.period_range(end=periodo_max, periods=36, freq="M")
    new_entries_all = cohort_df.groupby("periodo_cohorte").size().sort_index()
    s_new = new_entries_all.reindex(win)
    
    observed_months_sorted = new_entries_all.index.sort_values()
    if months_to_exclude > 0:
        months_to_drop = observed_months_sorted[:months_to_exclude]
        for m in months_to_drop:
            if m in s_new.index: s_new.loc[m] = np.nan

    avg_new_entries = (s_new.sum() / s_new.notna().sum()) if s_new.notna().sum() > 0 else np.nan

    rec_active_by_month = (cm[(cm["mes_vida"].between(12, 36)) & (cm["activo_mes"])]
                           .groupby("periodo_tx")["id"].nunique().reindex(win))
    avg_rec_active = (rec_active_by_month.sum() / rec_active_by_month.notna().sum()) if rec_active_by_month.notna().sum() > 0 else np.nan

    # Retención
    cohort_sizes = cohort_df.groupby("periodo_cohorte").size()
    rows_ret = []
    for k in range(0, 13): 
        eligible_coh = cohort_sizes.index[(cohort_sizes.index + k) <= periodo_max]
        eligible = float(cohort_sizes.loc[eligible_coh].sum()) if len(eligible_coh) else 0.0
        observed = float(cm[(cm["mes_vida"] == k) & (cm["activo_mes"])]["id"].nunique())
        
        if k == 0: ret = 1.0
        else: ret = (observed / eligible) if eligible > 0 else np.nan
        
        rows_ret.append((k, int(eligible), int(observed), ret * 100 if pd.notna(ret) else np.nan))

    ret_table = pd.DataFrame(rows_ret, columns=["mes_vida", "eligible_clients", "observed_active_clients", "retencion_%"])

    return {
        "raw": raw, "cohort_df": cohort_df, "cm": cm, "cm_arpu": cm_arpu,
        "ids_nuevos": ids_nuevos, "ids_recurrentes": ids_recurrentes,
        "periodo_max": periodo_max, "limite_outlier": limite_outlier,
        "outlier_customers": outlier_customers,
        "avg_new_entries": avg_new_entries, "avg_rec_active": avg_rec_active,
        "ret_table": ret_table, "s_new": s_new,
        "freq_new": freq_new, "freq_rec": freq_rec
    }

# ============================================================
# 4. Generación de Tablas y Reportes
# ============================================================
def tabla_por_mesvida(ids_base, rango_teorico, cm_arpu):
    seg_arpu = cm_arpu[cm_arpu["id"].isin(ids_base) & cm_arpu["mes_vida"].isin(rango_teorico)].copy()
    if seg_arpu.empty: return pd.DataFrame(), np.nan, np.nan

    arpu_den = seg_arpu[seg_arpu["activo_mes"]].groupby("mes_vida")["id"].nunique()
    arpu_num = seg_arpu.groupby("mes_vida")["monto_neto_cliente_mes"].sum()
    arpu = arpu_num / arpu_den.replace({0: np.nan})

    tabla = pd.DataFrame({
        "Clientes Activos": arpu_den,
        "Venta Total": arpu_num,
        "ARPU (Sin Outliers)": arpu
    }).reindex(list(rango_teorico))

    denom_total = arpu_den.sum()
    kpi_arpu = (arpu_num.sum() / denom_total) if denom_total > 0 else np.nan
    kpi_cli = arpu_den.dropna().mean() if len(arpu_den.dropna()) else np.nan
    return tabla.T, kpi_arpu, kpi_cli

def _b64_png_from_matplotlib(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _retencion_chart_png(ret_table: pd.DataFrame) -> str:
    x = np.asarray(ret_table["mes_vida"].values)
    y = np.asarray(ret_table["retencion_%"].values)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_title("Retención %")
    ax.set_xlabel("Mes de vida")
    ax.set_ylabel("Retención (%)")
    ax.grid(True, alpha=0.3)
    plt.close(fig)
    return _b64_png_from_matplotlib(fig)

def _df_html(df: pd.DataFrame, max_rows: int = 300) -> str:
    df2 = df.copy()
    if len(df2) > max_rows: df2 = df2.head(max_rows)
    return df2.to_html(border=0, float_format=lambda x: '{:,.0f}'.format(x))

def build_full_report_html(*, ids_nuevos_len, ids_recurrentes_len, outlier_customers_len,
                           limite_outlier, avg_new_entries, avg_rec_active,
                           t_new, t_rec, ret_table, yom_res, yom_bd) -> str:
    ret_png_b64 = _retencion_chart_png(ret_table)
    yom_html = "<p><i>No hay resultados YOM.</i></p>"
    if yom_res is not None:
        comp = yom_res.get("comparativa", pd.DataFrame()).copy()
        if not comp.empty:
            if "Cartera Activa" in comp.columns:
                comp["Cartera Activa"] = comp["Cartera Activa"].map(lambda x: f"{float(x):,.2f}")
            if "Ingreso Mensual" in comp.columns:
                comp["Ingreso Mensual"] = comp["Ingreso Mensual"].map(lambda x: f"${float(x):,.0f}")
            comp_html = comp.to_html(index=False, border=0)
        else: comp_html = ""

        yom_html = f"""
        <h2>🧮 Simulador YOM</h2>
        <ul>
          <li><b>Impacto Total Anual:</b> {fmt_money(yom_res.get("impacto_total_anual"))}</li>
          <li><b>Ingreso adicional (Nuevos):</b> {fmt_money(yom_res.get("ingreso_mensual_adic_nuevos"))}</li>
          <li><b>Ingreso adicional (Actuales):</b> {fmt_money(yom_res.get("ingreso_mensual_adic_actuales"))}</li>
        </ul>
        <h3>Comparativa</h3>
        {comp_html}
        """

    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Reporte Flux Analytics</title>
      <style>
        body {{ font-family: sans-serif; padding: 24px; }}
        h1, h2, h3 {{ margin: 10px 0; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
        .card {{ border: 1px solid #ddd; padding: 12px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background: #f3f3f3; }}
        img {{ max-width: 100%; }}
      </style>
    </head>
    <body>
      <h1>📊 Reporte Flux Analytics</h1>
      <div class="grid">
        <div class="card"><b>Base Nuevos</b><div>{ids_nuevos_len:,}</div></div>
        <div class="card"><b>Base Recurrentes</b><div>{ids_recurrentes_len:,}</div></div>
        <div class="card"><b>Outliers</b><div>{outlier_customers_len:,}</div></div>
        <div class="card"><b>Umbral</b><div>{fmt_money(limite_outlier)}</div></div>
      </div>
      <h2>🚀 NUEVOS</h2> {_df_html(t_new)}
      <h2>💎 RECURRENTES</h2> {_df_html(t_rec)}
      <h2>📈 RETENCIÓN</h2> {_df_html(ret_table)}
      <img src="data:image/png;base64,{ret_png_b64}" />
      {yom_html}
    </body>
    </html>
    """
    return html
