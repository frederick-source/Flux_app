import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ============================================================
# ✅ DASHBOARD: Cohortes (Nuevos 0–11 / Recurrentes 12–36)
# ============================================================

st.set_page_config(page_title="Cohortes Inteligente", layout="wide")
st.title("📊 Análisis de Cohortes (Nuevos 0–11 / Recurrentes 12–36)")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros")
outlier_pct = st.sidebar.slider("Corte Outliers (Top % en compras >0)", 0.01, 0.10, 0.05, 0.01)

# ✅ CAMBIO 1: Aumenté el max_value a 12 para darte más libertad de eliminar meses
months_to_exclude = st.sidebar.slider(
    "Meses a excluir al inicio (Censura Izquierda)",
    min_value=0,
    max_value=12, 
    value=1,
    help="Excluye los primeros N meses del cálculo de promedios para evitar datos inflados por la carga inicial histórica."
)

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is None:
    st.info("Sube un CSV para comenzar.")
    st.stop()

df = pd.read_csv(uploaded_file, low_memory=False)
df.columns = [str(c).strip() for c in df.columns]

st.subheader("🛠️ Mapeo de Columnas")
c1, c2, c3 = st.columns(3)

idx_date = next((i for i, c in enumerate(df.columns) if "fech" in c.lower() or "date" in c.lower()), 0)
idx_id   = next((i for i, c in enumerate(df.columns) if "id" in c.lower() or "rut" in c.lower()), 0)
idx_amt  = next((i for i, c in enumerate(df.columns) if "monto" in c.lower() or "sale" in c.lower() or "venta" in c.lower()), 0)

col_fecha = c1.selectbox("Fecha", df.columns, index=idx_date)
col_id    = c2.selectbox("ID Cliente", df.columns, index=idx_id)
col_monto = c3.selectbox("Monto/Venta", df.columns, index=idx_amt)

if not st.button("🚀 Generar Informe"):
    st.stop()

with st.spinner("Procesando..."):

    # 1) Limpieza base
    raw = df.rename(columns={col_fecha: "fecha", col_id: "id", col_monto: "monto"}).copy()
    raw["fecha"] = pd.to_datetime(raw["fecha"], errors="coerce")
    raw["id"] = raw["id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    raw["monto"] = parse_amount_series(raw["monto"])

    raw = raw.dropna(subset=["fecha", "id", "monto"]).copy()
    raw["periodo_tx"] = raw["fecha"].dt.to_period("M")

    fecha_max = raw["fecha"].max()
    periodo_max = fecha_max.to_period("M")

    # 2) Cohorte EXACTA
    first_pos = raw.loc[raw["monto"] > 0].groupby("id")["fecha"].min()
    cohort_df = first_pos.to_frame(name="fecha_cohorte")
    cohort_df["periodo_cohorte"] = cohort_df["fecha_cohorte"].dt.to_period("M")
    cohort_df["edad_actual_meses"] = cohort_df["periodo_cohorte"].apply(lambda p: months_diff(periodo_max, p))

    raw = raw[raw["id"].isin(cohort_df.index)].copy()

    # 3) Outliers
    tx_pos = raw.loc[raw["monto"] > 0, "monto"]
    limite_outlier = float(tx_pos.quantile(1.0 - outlier_pct)) if len(tx_pos) else np.nan
    raw["es_outlier_tx"] = (raw["monto"] > 0) & (raw["monto"] >= limite_outlier if pd.notna(limite_outlier) else False)
    outlier_customers = set(raw.loc[raw["es_outlier_tx"], "id"].unique())

    # 4) Cliente-mes
    raw = raw.join(cohort_df[["periodo_cohorte"]], on="id")
    cm = (raw.groupby(["id", "periodo_tx", "periodo_cohorte"], as_index=False)
            .agg(monto_neto_cliente_mes=("monto", "sum"), has_outlier_mes=("es_outlier_tx", "any")))

    cm["mes_vida"] = cm.apply(lambda r: months_diff(r["periodo_tx"], r["periodo_cohorte"]), axis=1)
    cm = cm[cm["mes_vida"] >= 0].copy()
    cm["activo_mes"] = cm["monto_neto_cliente_mes"] > 0

    cm_arpu = cm[~cm["has_outlier_mes"]].copy()
    cm_arpu["activo_mes"] = cm_arpu["monto_neto_cliente_mes"] > 0

    # 5) Segmentos
    ids_nuevos = cohort_df[cohort_df["edad_actual_meses"].between(0, 11)].index
    ids_recurrentes = cohort_df[cohort_df["edad_actual_meses"].between(12, 36)].index

    # 6) Tabla helper
    def tabla_por_mesvida(ids_base, rango_teorico):
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

    # 7) KPIs Globales
    win = pd.period_range(end=periodo_max, periods=36, freq="M")
    new_entries_all = cohort_df.groupby("periodo_cohorte").size().sort_index()
    s_new = new_entries_all.reindex(win)
    observed_months_sorted = new_entries_all.index.sort_values()

    # Aplica filtro de meses excluidos
    if months_to_exclude > 0:
        months_to_drop = observed_months_sorted[:months_to_exclude]
        for m in months_to_drop:
            if m in s_new.index: s_new.loc[m] = np.nan

    avg_new_entries = (s_new.sum() / s_new.notna().sum()) if s_new.notna().sum() > 0 else np.nan

    rec_active_by_month = (cm[(cm["mes_vida"].between(12, 36)) & (cm["activo_mes"])]
                           .groupby("periodo_tx")["id"].nunique().reindex(win))
    avg_rec_active = (rec_active_by_month.sum() / rec_active_by_month.notna().sum()) if rec_active_by_month.notna().sum() > 0 else np.nan

    # 8) Retención % (CORREGIDA PARA 100% EN MES 0)
    cohort_sizes = cohort_df.groupby("periodo_cohorte").size()
    rows_ret = []
    for k in range(0, 12):
        eligible_coh = cohort_sizes.index[(cohort_sizes.index + k) <= periodo_max]
        eligible = float(cohort_sizes.loc[eligible_coh].sum()) if len(eligible_coh) else 0.0
        observed = float(cm[(cm["mes_vida"] == k) & (cm["activo_mes"])]["id"].nunique())
        
        # ✅ CAMBIO 2: Lógica para forzar 100% en Mes 0
        if k == 0:
            ret = 1.0
        else:
            ret = (observed / eligible) if eligible > 0 else np.nan

        rows_ret.append((k, int(eligible), int(observed), ret * 100 if pd.notna(ret) else np.nan))

    ret_table = pd.DataFrame(rows_ret, columns=["mes_vida", "eligible_clients", "observed_active_clients", "retencion_%"])

    # 9) UI
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Base Nuevos (cohorte edad 0–11)", f"{len(ids_nuevos):,}")
    k2.metric("Base Recurrentes (cohorte edad 12–36)", f"{len(ids_recurrentes):,}")
    k3.metric("Clientes con ≥1 outlier tx", f"{len(outlier_customers):,}")
    k4.metric("Umbral outlier", fmt_money(limite_outlier))

    st.divider()
    cA, cB = st.columns(2)
    cA.metric("Nuevos que entran/mes (1ª compra positiva, 36m)", fmt_int(avg_new_entries))
    with cA.popover("Detalle filtro meses"):
        if months_to_exclude > 0:
            st.warning(f"⚠️ Se excluyeron los primeros {months_to_exclude} meses.")
        else:
            st.info("ℹ️ No se está excluyendo ningún mes inicial.")
    
    cB.metric("Recurrentes activos/mes (12–36m, 36m)", fmt_int(avg_rec_active))

    # 10) Tabs
    tab1, tab2, tab3 = st.tabs(["🚀 NUEVOS (0–11)", "💎 RECURRENTES (12–36)", "📈 RETENCIÓN"])

    with tab1:
        t_new, arpu_n, _ = tabla_por_mesvida(ids_nuevos, range(0, 12))
        m1, m2 = st.columns(2)
        m1.metric("Nuevos que entran/mes", fmt_int(avg_new_entries))
        with m1.popover("Info"):
            st.write("Promedio mensual de 1ª compra positiva.")
            # ✅ CAMBIO 3: Corregido nombre de variable para evitar error
            if months_to_exclude > 0:
                st.write(f"Se excluyen {months_to_exclude} meses iniciales.")

        m2.metric("ARPU (Sin Outliers)", fmt_money(arpu_n))
        if not t_new.empty:
            st.dataframe(t_new, use_container_width=True)
            with st.expander("Ver serie mensual"):
                st.dataframe(pd.DataFrame({"periodo": s_new.index.astype(str), "nuevos": s_new.values}), use_container_width=True)

    with tab2:
        t_rec, arpu_a, cli_a = tabla_por_mesvida(ids_recurrentes, range(12, 37))
        m1, m2 = st.columns(2)
        m1.metric("Recurrentes activos/mes", fmt_int(cli_a))
        m2.metric("ARPU (Sin Outliers)", fmt_money(arpu_a))
        if not t_rec.empty:
            st.dataframe(t_rec, use_container_width=True)

    with tab3:
        st.subheader("Retención % (denominador = elegibles)")
        st.dataframe(ret_table, use_container_width=True)
        st.line_chart(ret_table.set_index("mes_vida")[["retencion_%"]])
