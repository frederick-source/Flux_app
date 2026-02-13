import streamlit as st
import pandas as pd
import numpy as np
import flux_backend as backend
import base64
import os

# ============================================================
# 1. CONFIGURACIÓN VISUAL
# ============================================================
st.set_page_config(
    page_title="Flux Analytics | Powered by Yom",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_html = ""
if os.path.exists("logo.png"):
    try:
        img_b64 = get_img_as_base64("logo.png")
        logo_html = f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{img_b64}" style="max-width: 80%; max-height: 80px;">
            </div>
        """
    except:
        logo_html = "<h1 style='color:white;text-align:center;'>YOM</h1>"
else:
    logo_html = """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: white; font-size: 3rem; margin:0;">YOM</h1>
        </div>
    """

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background-color: #FFFFFF; color: #1F2937; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #4C1D95 0%, #2E1065 100%); }
    section[data-testid="stSidebar"] * { color: #F3F4F6 !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] div { color: #1F2937 !important; -webkit-text-fill-color: #1F2937 !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #1F2937 !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] { background-color: white !important; }
    section[data-testid="stSidebar"] input { color: #1F2937 !important; background-color: white !important; }
    section[data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] { background-color: white !important; }
    h1 { background: linear-gradient(90deg, #7C3AED 0%, #DB2777 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; }
    h2, h3 { color: #4C1D95 !important; font-weight: 700 !important; }
    div[data-testid="stMetric"] { background-color: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(124, 58, 237, 0.1); }
    div[data-testid="stMetricLabel"] { color: #6D28D9 !important; }
    div[data-testid="stMetricValue"] { color: #111827 !important; }
    .stButton > button { background: linear-gradient(90deg, #7C3AED, #9333EA); color: white; border: none; border-radius: 8px; font-weight: 600; }
    [data-testid="stFileUploaderDropzone"] { background-color: rgba(255, 255, 255, 0.1) !important; border: 1px dashed #A78BFA !important; }
    [data-testid="stFileUploaderDropzone"] div { color: #E5E7EB !important; }
    [data-testid="stFileUploaderDropzone"] button { background: #6D28D9 !important; color: white !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

# Helper para formatear dinero estilo chileno (1.000.000)
def fmt_clp(x):
    if pd.isna(x) or x == 0: return "-"
    return f"${x:,.0f}".replace(",", ".")

# ============================================================
# 2. BARRA LATERAL
# ============================================================
with st.sidebar:
    st.markdown(logo_html, unsafe_allow_html=True)
    st.markdown("---")
    st.header("⚙️ Configuración")
    outlier_pct = st.slider("Corte Outliers (Top %)", 0.01, 0.10, 0.05, 0.01)
    months_to_exclude = st.slider("Meses a excluir (Inicio)", 0, 12, 1)
    st.markdown("---")
    st.markdown("### 📂 Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu CSV aquí", type=["csv"])

if uploaded_file is None:
    st.markdown("""
        <div style="text-align: center; padding: 60px 20px; margin-top: 50px; background: linear-gradient(135deg, #F5F3FF 0%, #FFFFFF 100%); border-radius: 20px; border: 1px solid #DDD6FE; box-shadow: 0 10px 30px -10px rgba(124, 58, 237, 0.15);">
            <h1 style="font-size: 4rem; margin-bottom: 15px; line-height: 1.1;">¡Bienvenido a Flux!</h1>
            <h3 style="color: #6D28D9 !important; font-weight: 500 !important; font-size: 1.6rem; margin-bottom: 30px;">
                Tecnología de <b>YOM</b> para potenciar tus análisis de ventas.
            </h3>
            <p style="color: #4B5563; font-size: 1.2rem; max-width: 700px; margin: 0 auto; line-height: 1.6;">
                Sube tu archivo en el panel izquierdo para comenzar.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

file_bytes = uploaded_file.getvalue()
try:
    df_raw_load = backend.load_csv(file_bytes)
    df_raw_load.columns = [str(c).strip() for c in df_raw_load.columns]
except Exception as e:
    st.error(f"Error leyendo archivo: {e}")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🛠️ Columnas Detectadas")
    idx_date = next((i for i, c in enumerate(df_raw_load.columns) if "fech" in c.lower() or "date" in c.lower()), 0)
    idx_id    = next((i for i, c in enumerate(df_raw_load.columns) if "id" in c.lower() or "rut" in c.lower()), 0)
    idx_amt   = next((i for i, c in enumerate(df_raw_load.columns) if "monto" in c.lower() or "sale" in c.lower() or "venta" in c.lower()), 0)
    col_fecha = st.selectbox("Fecha", df_raw_load.columns, index=idx_date)
    col_id    = st.selectbox("ID Cliente", df_raw_load.columns, index=idx_id)
    col_monto = st.selectbox("Monto", df_raw_load.columns, index=idx_amt)
    run_btn = st.button("🚀 Actualizar Análisis", type="primary")

if "report_ready" not in st.session_state:
    st.session_state["report_ready"] = True

with st.spinner("🔮 Procesando con Flux Core..."):
    try:
        df_clean = backend.clean_input_data(df_raw_load, col_fecha, col_id, col_monto)
        R = backend.compute_historico(df_clean, "fecha", "id", "monto", outlier_pct, months_to_exclude)
    except Exception as e:
        st.error(f"Error en cálculo: {e}")
        st.stop()

raw, cohort_df, cm, cm_arpu = R["raw"], R["cohort_df"], R["cm"], R["cm_arpu"]
ids_nuevos, ids_recurrentes = R["ids_nuevos"], R["ids_recurrentes"]
periodo_max, limite_outlier = R["periodo_max"], R["limite_outlier"]
outlier_customers = R["outlier_customers"]
avg_new_entries, avg_rec_active = R["avg_new_entries"], R["avg_rec_active"]
ret_table, s_new = R["ret_table"], R["s_new"]
freq_new, freq_rec = R["freq_new"], R["freq_rec"]

st.title("Flux Analytics Dashboard")
st.markdown(f"**Corte:** {periodo_max} | **Clientes:** {len(cohort_df):,}")
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Base Nuevos (0-11m)", f"{len(ids_nuevos):,}")
k2.metric("Base Recurrentes (12-36m)", f"{len(ids_recurrentes):,}")
k3.metric("Clientes Outliers", f"{len(outlier_customers):,}")
k4.metric("Umbral Outlier", backend.fmt_money(limite_outlier))
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 NUEVOS", "💎 RECURRENTES", "📈 RETENCIÓN", "🔮 SIMULADOR YOM"])

with tab1:
    st.markdown("### Performance Adquisición")
    _, arpu_n_real, _ = backend.tabla_por_mesvida(ids_nuevos, range(0, 12), cm_arpu)
    m1, m2, m3 = st.columns(3)
    m1.metric("Nuevos/Mes", backend.fmt_int(avg_new_entries))
    m2.metric("ARPU Nuevos", backend.fmt_money(arpu_n_real))
    m3.metric("Frecuencia", f"{freq_new:.2f} tx/mes")
    ids_new_viz = cohort_df[cohort_df["edad_actual_meses"].between(0, 11)].index
    t_new, _, _ = backend.tabla_por_mesvida(ids_new_viz, range(0, 12), cm_arpu)
    if not t_new.empty:
        st.dataframe(t_new.applymap(lambda x: "{:,.0f}".format(x)), use_container_width=True)

with tab2:
    st.markdown("### Salud Cartera")
    _, arpu_a_real, cli_a_real = backend.tabla_por_mesvida(ids_recurrentes, range(12, 37), cm_arpu)
    m1, m2, m3 = st.columns(3)
    m1.metric("Recurrentes Activos", backend.fmt_int(cli_a_real))
    m2.metric("ARPU Recurrentes", backend.fmt_money(arpu_a_real))
    m3.metric("Frecuencia", f"{freq_rec:.2f} tx/mes")
    ids_rec_viz = cohort_df[cohort_df["edad_actual_meses"].between(12, 37)].index
    t_rec, _, _ = backend.tabla_por_mesvida(ids_rec_viz, range(12, 38), cm_arpu)
    if not t_rec.empty:
        st.dataframe(t_rec.applymap(lambda x: "{:,.0f}".format(x)), use_container_width=True)

with tab3:
    st.markdown("### Retención")
    ret_view = ret_table.copy()
    ret_view.columns = ["Mes", "Elegibles", "Activos", "Retención %"]
    ret_view["Retención %"] = ret_view["Retención %"].apply(lambda x: "{:.1f}%".format(x) if pd.notnull(x) else "")
    c1, c2 = st.columns([2, 1])
    c1.line_chart(ret_table.set_index("mes_vida")["retencion_%"])
    c2.dataframe(ret_view, use_container_width=True, height=400)

# ============================================================
# TAB 4: SIMULADOR YOM (SEPARADO Y FORMATEADO)
# ============================================================
with tab4:
    st.header("🔮 Simulador de Impacto YOM")
    st.markdown("Calculadora de capacidad comercial y proyección de impacto por retención.")

    # --- 1. DATOS BASE ---
    try:
        _, arpu_n_yom, _ = backend.tabla_por_mesvida(ids_nuevos, range(0, 12), cm_arpu)
        freq_compra_val = freq_new if freq_new > 0 else 1.7
        val_arpu_n = float(arpu_n_yom) if pd.notna(arpu_n_yom) else 4800000.0
        val_cli_nuevos = float(avg_new_entries) if pd.notna(avg_new_entries) else 14.0
    except:
        val_arpu_n = 4800000.0
        val_cli_nuevos = 14.0
        freq_compra_val = 1.7

    # --- 2. INPUTS Y ESCENARIO ---
    st.subheader("1. Configuración del Escenario")
    col_in_1, col_in_2, col_in_3, col_in_4 = st.columns(4)
    visitas_dia = col_in_1.number_input("Visitas Vendedor/Día", value=22)
    n_vendedores = col_in_2.number_input("N° Vendedores", value=2)
    tasa_dig = col_in_3.number_input("Tasa Digitalización (%)", value=15.0) / 100.0
    gestiones_react = col_in_4.number_input("Gestiones p/ Reactivación", value=4.0)
    col_in_5, col_in_6, col_in_7, col_in_8 = st.columns(4)
    foco_act = col_in_5.number_input("Foco en Nuevos (%)", value=10.0) / 100.0
    delta_retencion = col_in_6.number_input("Aumento Lealtad/Retención (%)", value=2.5, step=0.1) / 100.0
    dias_mes = 22
    
    # Cálculos a-p
    val_a = val_arpu_n
    val_b = val_cli_nuevos
    val_c = visitas_dia
    val_d = dias_mes
    val_e = val_c * val_d
    val_f = tasa_dig
    val_g = val_e * val_f
    val_h = gestiones_react
    val_i = val_g / val_h if val_h > 0 else 0
    val_j = n_vendedores
    val_k = val_i * val_j
    val_l = foco_act
    val_m = round(val_k * val_l) # Capacidad Nuevos (Clientes Incrementales)
    val_n = round(val_k * (1 - val_l)) # Capacidad Resurrección
    val_o = freq_compra_val
    val_p = val_a / val_o if val_o > 0 else 0

    with st.expander("📋 Ver Tabla de Escenario (Inputs Calculados)", expanded=True):
        df_escenario = pd.DataFrame([
            {"Concepto": "a) ARPU Nuevos ($)", "Valor": fmt_clp(val_a)},
            {"Concepto": "b) Clientes Nuevos/Mes (Base)", "Valor": f"{val_b:.1f}"},
            {"Concepto": "c) Visitas/Vendedor/Día", "Valor": f"{val_c:.1f}"},
            {"Concepto": "d) Días Laborales", "Valor": f"{val_d}"},
            {"Concepto": "e) Cap. Gestión Mensual (c*d)", "Valor": f"{val_e:.1f}"},
            {"Concepto": "f) Tasa Digitalización", "Valor": f"{val_f*100:.1f}%"},
            {"Concepto": "g) Nueva Capacidad (e*f)", "Valor": f"{val_g:.1f}"},
            {"Concepto": "h) Gestiones p/ Reactivación", "Valor": f"{val_h:.1f}"},
            {"Concepto": "i) Nueva Cap. Única (g/h)", "Valor": f"{val_i:.1f}"},
            {"Concepto": "j) N° Vendedores", "Valor": f"{val_j}"},
            {"Concepto": "k) Total Mensual Cap. Única (i*j)", "Valor": f"{val_k:.1f}"},
            {"Concepto": "l) Foco Activación Nuevos", "Valor": f"{val_l*100:.1f}%"},
            {"Concepto": "m) Cap. Clientes Nuevos (Base Incrementales)", "Valor": f"🚀 {val_m:.0f}"},
            {"Concepto": "n) Cap. Clientes Resurrección (Base Resurrección)", "Valor": f"♻️ {val_n:.0f}"},
            {"Concepto": "o) Frecuencia de Compra", "Valor": f"{val_o:.2f}"},
            {"Concepto": "p) Ticket Promedio (AOV)", "Valor": fmt_clp(val_p)},
        ])
        st.dataframe(df_escenario, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # =========================================================
    # SECCIÓN 3: ANÁLISIS ORGÁNICO
    # =========================================================
    st.subheader(f"2. Crecimiento Orgánico (Retención +{delta_retencion*100:.1f}%)")
    
    # Cálculo
    clientes_extra_cohorte = np.ceil(val_b * delta_retencion)
    meses_proyeccion = 12
    cols_meses = list(range(1, meses_proyeccion + 1))
    idx_cohortes = [f"Cohorte {x+1}" for x in range(meses_proyeccion)]
    
    matriz_org_impacto = np.zeros((meses_proyeccion, meses_proyeccion))
    matriz_org_clientes = np.zeros((meses_proyeccion, meses_proyeccion))
    
    for i in range(meses_proyeccion):
        for mes_vida in range(meses_proyeccion):
            idx_calendario = i + mes_vida
            if idx_calendario < meses_proyeccion:
                if mes_vida == 0:
                    matriz_org_impacto[i, idx_calendario] = 0
                    matriz_org_clientes[i, idx_calendario] = 0
                else:
                    matriz_org_impacto[i, idx_calendario] = clientes_extra_cohorte * val_a
                    matriz_org_clientes[i, idx_calendario] = clientes_extra_cohorte

    # KPIs Orgánicos
    df_organico = pd.DataFrame(matriz_org_impacto, columns=cols_meses, index=idx_cohortes)
    df_organico_clientes = pd.DataFrame(matriz_org_clientes, columns=cols_meses, index=idx_cohortes)
    
    total_org_anual = df_organico.sum().sum()
    total_org_cli_12 = matriz_org_clientes[:, 11].sum()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Impacto Orgánico Anual ($)", fmt_clp(total_org_anual))
    k2.metric("+Clientes Activos (Mes 12)", f"{int(total_org_cli_12)}")
    k3.metric("Clientes Salvados/Cohorte", f"{int(clientes_extra_cohorte)}")
    
    # Gráfico Solo Orgánico
    st.bar_chart(df_organico.sum(axis=0))
    
    # Tablas Orgánicas Separadas
    tab_org_1, tab_org_2 = st.tabs(["💵 Ingresos ($)", "👥 Clientes (#)"])
    with tab_org_1:
        st.caption("Dinero extra generado por cohorte (acumulativo)")
        st.dataframe(df_organico.applymap(fmt_clp), use_container_width=True)
    with tab_org_2:
        st.caption("Clientes 'enteros' salvados que permanecen activos")
        st.dataframe(df_organico_clientes.applymap(lambda x: f"{int(x)}" if x > 0 else "-"), use_container_width=True)
        
    st.markdown("---")

    # =========================================================
    # SECCIÓN 4: ANÁLISIS INCREMENTALES (ACTIVACIÓN)
    # =========================================================
    st.subheader(f"3. Clientes Incrementales (Activación Comercial)")
    
    # Cálculo
    curva_retencion_base = [1.0, 0.141, 0.149, 0.149, 0.155, 0.165, 0.154, 0.172, 0.168, 0.170, 0.172, 0.185]
    matriz_inc_impacto = np.zeros((meses_proyeccion, meses_proyeccion))
    matriz_inc_clientes = np.zeros((meses_proyeccion, meses_proyeccion))
    
    for i in range(meses_proyeccion):
        for mes_vida in range(meses_proyeccion):
            idx_calendario = i + mes_vida
            if idx_calendario < meses_proyeccion:
                idx_ret = mes_vida if mes_vida < len(curva_retencion_base) else -1
                tasa_ret_base = curva_retencion_base[idx_ret]
                
                if mes_vida == 0:
                    clientes_inc = val_m # Boost inicial (Input 'm')
                else:
                    tasa_ret_nueva = tasa_ret_base + delta_retencion
                    clientes_inc = np.round(val_m * tasa_ret_nueva)
                
                matriz_inc_impacto[i, idx_calendario] = clientes_inc * val_a
                matriz_inc_clientes[i, idx_calendario] = clientes_inc
    
    # KPIs Incrementales
    df_incremental = pd.DataFrame(matriz_inc_impacto, columns=cols_meses, index=idx_cohortes)
    df_incremental_clientes = pd.DataFrame(matriz_inc_clientes, columns=cols_meses, index=idx_cohortes)
    
    total_inc_anual = df_incremental.sum().sum()
    total_foco_act = val_m * 12
    
    k1, k2 = st.columns(2)
    k1.metric("Impacto Incremental Anual ($)", fmt_clp(total_inc_anual))
    k2.metric("+Foco Activación (Clientes Totales)", f"{int(total_foco_act)}")
    
    # Gráfico Solo Incremental
    st.bar_chart(df_incremental.sum(axis=0))

    # Tablas Incrementales Separadas
    tab_inc_1, tab_inc_2 = st.tabs(["💵 Ingresos ($)", "👥 Clientes (#)"])
    with tab_inc_1:
        st.caption("Ingresos generados por campañas de activación")
        st.dataframe(df_incremental.applymap(fmt_clp), use_container_width=True)
    with tab_inc_2:
        st.caption("Evolución de clientes activados (Cohorte inicial vs. Retenidos)")
        st.dataframe(df_incremental_clientes.applymap(lambda x: f"{int(x)}" if x > 0 else "-"), use_container_width=True)

# Descarga del reporte
st.markdown("---")
if uploaded_file is not None:
    html = backend.build_full_report_html(
        ids_nuevos_len=len(ids_nuevos), ids_recurrentes_len=len(ids_recurrentes),
        outlier_customers_len=len(outlier_customers), limite_outlier=limite_outlier,
        avg_new_entries=avg_new_entries, avg_rec_active=avg_rec_active,
        t_new=t_new, t_rec=t_rec, ret_table=ret_table, yom_res=None, yom_bd=None
    )
    st.download_button("📥 Descargar Reporte Completo", html.encode("utf-8"), "Flux_Yom.html", "text/html")
