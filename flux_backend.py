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
# 2. Carga y Limpieza
# ============================================================
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    sample = file_bytes[:10000].decode('utf-8', errors='ignore')
    count_commas = sample.count(',')
    count_semicolons = sample.count(';')
    detected_sep = ';' if count_semicolons > count_commas else ','
    
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=detected_sep, low_memory=False)
        return df
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python', encoding='latin1')

def clean_input_data(df: pd.DataFrame, col_fecha: str, col_id: str, col_monto: str) -> pd.DataFrame:
    clean = df[[col_fecha, col_id, col_monto]].copy()
    clean.columns = ["fecha", "id", "monto"]

    clean["id"] = clean["id"].astype(str).str.strip()
    clean["id"] = clean["id"].str.split(pat=" ", n=1).str[0]
    clean["id"] = clean["id"].str.replace(r"[^a-zA-Z0-9\-\.]", "", regex=True)

    clean["monto"] = parse_amount_series(clean["monto"])
    
    # --- CORRECCIÓN AQUÍ: Lectura de fechas más flexible ---
    # Eliminamos dayfirst=True forzado para permitir formatos YYYY-M-D (como el de tu archivo)
    # Primero intentamos la detección estándar (funciona mejor con YYYY-MM-DD)
    clean["fecha"] = pd.to_datetime(clean["fecha"], errors="coerce")
    
    # Si la detección falló (muchos NaT), intentamos forzando día primero (para casos DD/MM/YYYY)
    if clean["fecha"].isna().sum() > len(clean) * 0.5:
        clean["fecha"] = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")
    
    # Eliminamos solo lo que realmente no se pudo leer
    clean = clean.dropna(subset=["fecha", "monto"])

    return clean

# ============================================================
# 3. Cálculo de Cohortes
# ============================================================
@st.cache_data(show_spinner=False)
def compute_historico(df: pd.DataFrame, col_fecha: str, col_id: str, col_monto: str,
                     outlier_pct: float, months_to_exclude: int):
    raw = df.rename(columns={col_fecha: "fecha", col_id: "id", col_monto: "monto"}).copy()

    # Aquí también aseguramos una lectura limpia sin forzar parámetros extraños
    raw["fecha"] = pd.to_datetime(raw["fecha"], errors="coerce")
    raw["id"] = raw["id"].astype(str)
    raw["monto"] = parse_amount_series(raw["monto"])

    raw = raw.dropna(subset=["fecha", "id", "monto"]).copy()
    raw = raw[raw["monto"] > 0].copy()

    raw = (raw.groupby(["id", "fecha"], as_index=False)
           .agg(monto=("monto", "sum")))

    raw["periodo_tx"] = raw["fecha"].dt.to_period("M")
    fecha_max = raw["fecha"].max()
    periodo_max = fecha_max.to_period("M")

    first_pos = raw.groupby("id")["fecha"].min()
    cohort_df = first_pos.to_frame(name="fecha_cohorte")
    cohort_df["periodo_cohorte"] = cohort_df["fecha_cohorte"].dt.to_period("M")
    cohort_df["edad_actual_meses"] = cohort_df["periodo_cohorte"].apply(lambda p: months_diff(periodo_max, p))

    raw = raw[raw["id"].isin(cohort_df.index)].copy()

    tx_pos = raw["monto"]
    limite_outlier = float(tx_pos.quantile(1.0 - outlier_pct)) if len(tx_pos) else np.nan
    raw["es_outlier_tx"] = raw["monto"] >= limite_outlier if pd.notna(limite_outlier) else False
    outlier_customers = set(raw.loc[raw["es_outlier_tx"], "id"].unique())

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

    ids_nuevos = cohort_df[cohort_df["edad_actual_meses"].between(0, 11)].index
    ids_recurrentes = cohort_df[cohort_df["edad_actual_meses"].between(12, 36)].index

    mask_new = (cm["id"].isin(ids_nuevos)) & (cm["mes_vida"].between(0, 11)) & (cm["activo_mes"])
    freq_new = cm.loc[mask_new, "tx_count"].mean() if mask_new.any() else 0.0

    mask_rec = (cm["id"].isin(ids_recurrentes)) & (cm["mes_vida"].between(12, 36)) & (cm["activo_mes"])
    freq_rec = cm.loc[mask_rec, "tx_count"].mean() if mask_rec.any() else 0.0

    win = pd.period_range(end=periodo_max, periods=36, freq="M")
    new_entries_all = cohort_df.groupby("periodo_cohorte").size().sort_index()
    s_new = new_entries_all.reindex(win)

    observed_months_sorted = new_entries_all.index.sort_values()
    if months_to_exclude > 0:
        months_to_drop = observed_months_sorted[:months_to_exclude]
        for m in months_to_drop:
            if m in s_new.index: s_new.loc[m] = np.nan

    avg_new_entries = (s_new.sum() / s_new.notna().sum()) if s_new.notna().sum() > 0 else np.nan

    ventana_inicio = periodo_max - 5
    mask_active_window = (cm["activo_mes"]) & (cm["periodo_tx"] >= ventana_inicio) & (cm["periodo_tx"] <= periodo_max)
    ids_activos_recientes = cm.loc[mask_active_window, "id"].unique()
    ids_recurrentes_activos = np.intersect1d(ids_activos_recientes, ids_recurrentes)
    avg_rec_active = len(ids_recurrentes_activos)

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
# 4. Generación de Reportes HTML (ACTUALIZADO CON YOM)
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
    if ret_table is None or ret_table.empty: return ""
    x = np.asarray(ret_table["mes_vida"].values)
    y = np.asarray(ret_table["retencion_%"].values)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', color='#4C1D95', linewidth=2)
    ax.set_title("Curva de Retención (%)", fontsize=14)
    ax.set_xlabel("Mes de vida")
    ax.set_ylabel("Retención (%)")
    ax.grid(True, alpha=0.3)
    for i, txt in enumerate(y):
        if pd.notna(txt):
            ax.annotate(f"{txt:.1f}%", (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    encoded = _b64_png_from_matplotlib(fig)
    plt.close(fig)
    return encoded

def build_full_report_html(ids_nuevos_len, ids_recurrentes_len, outlier_customers_len, 
                           limite_outlier, avg_new_entries, avg_rec_active, 
                           t_new, t_rec, ret_table,
                           df_yom_new=None, df_yom_rec=None, yom_impacto_total=None):
    
    def format_val(x):
        if isinstance(x, (int, float)):
            if pd.isna(x): return "-"
            return f"{x:,.0f}".replace(",", ".")
        return x
    
    chart_b64 = _retencion_chart_png(ret_table)
    img_tag = f'<img src="data:image/png;base64,{chart_b64}" style="width:100%; max-width:800px; margin: 20px auto; display:block;" />' if chart_b64 else "<p>Sin datos</p>"

    html_new = t_new.applymap(format_val).to_html(classes='styled-table') if not t_new.empty else "<p>Sin datos</p>"
    html_rec = t_rec.applymap(format_val).to_html(classes='styled-table') if not t_rec.empty else "<p>Sin datos</p>"
    html_ret = ret_table.applymap(format_val).to_html(classes='styled-table', index=False) if not ret_table.empty else "<p>Sin datos</p>"

    # Renderizado de tablas YOM
    html_yom_n = df_yom_new.to_html(classes='styled-table', index=False) if df_yom_new is not None else "<p>Sin simulación activa</p>"
    html_yom_r = df_yom_rec.to_html(classes='styled-table', index=False) if df_yom_rec is not None else "<p>Sin simulación activa</p>"
    str_impacto = yom_impacto_total if yom_impacto_total else "$0"

    style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; color: #1F2937; padding: 40px; background-color: #F3F4F6; margin: 0; }
    .container { max-width: 1000px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    h1 { color: #4C1D95; border-bottom: 3px solid #7C3AED; padding-bottom: 10px; margin-bottom: 30px; }
    h2 { color: #6D28D9; margin-top: 40px; border-left: 5px solid #7C3AED; padding-left: 15px; font-size: 1.5rem; }
    h3 { color: #4C1D95; margin-top: 25px; }
    
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
    .kpi-card { background: #F5F3FF; padding: 20px; border-radius: 8px; border: 1px solid #DDD6FE; text-align: center; }
    .kpi-label { color: #6D28D9; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 800; color: #111827; }
    .kpi-sub { font-size: 0.8rem; color: #6B7280; margin-top: 5px; }

    .styled-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; box-shadow: 0 0 20px rgba(0, 0, 0, 0.05); }
    .styled-table thead tr { background-color: #4C1D95; color: #ffffff; text-align: center; }
    .styled-table th, .styled-table td { padding: 12px 15px; border: 1px solid #E5E7EB; }
    .styled-table tbody tr { border-bottom: 1px solid #dddddd; }
    .styled-table tbody tr:nth-of-type(even) { background-color: #F9FAFB; }
    .styled-table td { text-align: right; }
    .styled-table td:first-child { text-align: left; font-weight: 600; color: #374151; }
    
    .impact-box { background-color: #F0FDF4; border: 1px solid #BBF7D0; padding: 25px; border-radius: 12px; text-align: center; margin: 30px 0; }
    .impact-title { color: #15803D; font-weight: 700; text-transform: uppercase; font-size: 0.9rem; margin-bottom: 5px; }
    .impact-value { color: #14532D; font-size: 2.8rem; font-weight: 800; }
    .impact-sub { color: #166534; font-size: 0.9rem; opacity: 0.8; }

    .footer { margin-top: 60px; font-size: 0.85rem; color: #9CA3AF; text-align: center; border-top: 1px solid #E5E7EB; padding-top: 20px; }
    </style>
    """

    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Reporte Flux Analytics</title>
        {style}
    </head>
    <body>
    <div class="container">
        <div style="text-align:right; color:#9CA3AF; font-size:0.8rem;">{datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
        <h1>Flux Analytics | Informe Ejecutivo</h1>
        
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Clientes Nuevos</div>
                <div class="kpi-value">{format_val(ids_nuevos_len)}</div>
                <div class="kpi-sub">Total Histórico (0-11 m)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Clientes Recurrentes</div>
                <div class="kpi-value">{format_val(ids_recurrentes_len)}</div>
                <div class="kpi-sub">Total Histórico (12-36 m)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Entrada Mensual</div>
                <div class="kpi-value">{format_val(avg_new_entries)}</div>
                <div class="kpi-sub">Promedio Nuevos/Mes</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Base Activa</div>
                <div class="kpi-value">{format_val(avg_rec_active)}</div>
                <div class="kpi-sub">Únicos en últimos 6 meses</div>
            </div>
        </div>

        <h2>🚀 Análisis de Nuevos</h2>
        <p>Comportamiento de clientes captados en el último año.</p>
        {html_new}

        <h2>💎 Análisis de Recurrentes</h2>
        <p>Desempeño de la base fidelizada (antigüedad > 12 meses).</p>
        {html_rec}

        <h2>📈 Retención</h2>
        {img_tag}
        {html_ret}

        <h2>🔮 Simulación de Impacto YOM</h2>
        <div class="impact-box">
            <div class="impact-title">Impacto Total Mensual Proyectado</div>
            <div class="impact-value">{str_impacto}</div>
            <div class="impact-sub">Suma de eficiencia operativa y gestión comercial YOM</div>
        </div>

        <h3>A. Impacto en Nuevos Clientes (Adquisición & Recuperación)</h3>
        {html_yom_n}

        <h3>B. Impacto en Clientes Recurrentes (Cartera Actual)</h3>
        {html_yom_r}

        <div class="footer">
            Reporte generado por Flux Analytics Powered by YOM. Propiedad de Frederick Russell Krauss.
        </div>
    </div>
    </body>
    </html>
    """
    return html_template
