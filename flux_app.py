import streamlit as st
import pandas as pd
import numpy as np
import flux_backend as backend
import base64
import os
import re

# ============================================================
# 1. CONFIGURACIÓN VISUAL
# ============================================================
st.set_page_config(
    page_title="Flux Analytics | Powered by Yom",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background-color: #FFFFFF; color: #1F2937; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #4C1D95 0%, #2E1065 100%); }
    section[data-testid="stSidebar"] * { color: #F3F4F6 !important; }
    
    /* Inputs Sidebar */
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

def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def fmt_clp(x):
    if pd.isna(x) or x == 0: return "-"
    return f"${x:,.0f}".replace(",", ".")

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

# ============================================================
# 2. BARRA LATERAL
# ============================================================
with st.sidebar:
    st.markdown(logo_html, unsafe_allow_html=True)
    st.markdown("---")
    st.header("⚙️ Configuración")
    
    st.info("Ajustes de limpieza de datos.")
    outlier_pct = st.slider("Corte Outliers (Top %)", 0.01, 0.10, 0.05, 0.01, help="Elimina el % superior de ventas para no distorsionar los promedios con 'ballenas'.")
    months_to_exclude = st.slider("Meses a excluir (Inicio)", 0, 12, 1, help="Ignora los primeros meses si la data histórica no es confiable.")
    
    st.markdown("---")
    st.markdown("### 📂 Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu CSV aquí", type=["csv"])
    
    st.markdown("---")
    st.caption("Desarrollado con cariño por tu practicante favorito de YOM (Frederick Russell).")

# ============================================================
# PANTALLA DE BIENVENIDA (INSTRUCTIVO)
# ============================================================
if uploaded_file is None:
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("¡Bienvenido a Flux!")
        st.subheader("Analytics Powered by YOM 🚀")
        st.markdown("---")
        
        st.markdown("### 📋 Instrucciones para tu archivo")
        st.info("""
        Para que el análisis funcione perfecto, tu archivo CSV debería tener preferentemente estas 3 columnas:
        
        1. **sales**: El monto de la venta (ej: 15000).
        2. **date**: La fecha de la transacción (ej: 2024-01-30).
        3. **id_cliente**: Un identificador único (RUT, email, ID).
        """)
        st.caption("💡 *Si tu archivo tiene otros nombres, no te preocupes. El sistema es inteligente e intentará leerlo igual, pero seguir el formato mejora la precisión.*")
        
        st.warning("👈 **Para comenzar, sube tu archivo en el menú de la izquierda.**")
    st.stop()

# ============================================================
# PROCESAMIENTO
# ============================================================
file_bytes = uploaded_file.getvalue()
try:
    df_raw_load = backend.load_csv(file_bytes)
    df_raw_load.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c).strip()) for c in df_raw_load.columns]
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
        
        # VENTAS ANUALES
        df_clean['year'] = df_clean['fecha'].dt.year
        sales_per_year = df_clean.groupby('year')['monto'].sum().sort_index(ascending=True)
        last_3_years = sales_per_year.tail(3)
        
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

# ============================================================
# DASHBOARD HEADER
# ============================================================
st.title("Flux Analytics Dashboard")
st.markdown(f"**Corte:** {periodo_max} | **Clientes:** {len(cohort_df):,}")
st.markdown("---")

if not last_3_years.empty:
    st.markdown("### 🏆 Ventas Totales por Año")
    cols_annual = st.columns(len(last_3_years))
    for i, (year, total) in enumerate(last_3_years.items()):
        cols_annual[i].metric(f"Ventas {year}", fmt_clp(total), help=f"Venta bruta total encontrada en el archivo para el año {year}.")
    st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Base Nuevos", f"{len(ids_nuevos):,}", help="Clientes que hicieron su primera compra hace menos de 12 meses.")
k2.metric("Base Recurrentes", f"{len(ids_recurrentes):,}", help="Clientes activos con antigüedad entre 12 y 36 meses.")
k3.metric("Clientes Outliers", f"{len(outlier_customers):,}", help=f"Clientes excluidos del análisis por tener un volumen de compra atípico (Top {outlier_pct*100}%).")
k4.metric("Umbral Outlier", backend.fmt_money(limite_outlier), help="Monto de corte para considerar una venta como 'normal'.")
st.markdown("---")

# ============================================================
# TABS CON EXPLICACIONES
# ============================================================
tab_intro, tab1, tab2, tab3, tab4 = st.tabs(["🏠 INICIO", "🚀 NUEVOS", "💎 RECURRENTES", "📈 RETENCIÓN", "🔮 SIMULADOR YOM"])

# --- TAB 0: INTRODUCCIÓN ---
with tab_intro:
    st.header("Bienvenido a Flux Analytics")
    st.markdown("""
    Esta herramienta ha sido diseñada para transformar tus datos de ventas en inteligencia comercial accionable. 
    Analizamos tu archivo para segmentar clientes, entender su retención y proyectar escenarios de crecimiento.
    """)
    
    st.info("""
    **¿Qué encontrarás aquí?**
    * **Diagnóstico:** Entiende cuántos clientes nuevos estás captando y cuántos realmente se quedan (Recurrentes).
    * **Retención:** Visualiza la "fuga" de clientes mes a mes.
    * **Simulador YOM:** Una calculadora financiera que proyecta cuánto dinero extra puedes ganar activando las palancas de crecimiento de YOM.
    """)
    
    st.markdown("### 📌 Ficha Técnica")
    st.markdown("""
    * **Código Fuente:** Repositorio privado en **GitHub**.
    * **Propiedad:** Compartido a las cuentas de **Diego Fuentealba**.
    * **Infraestructura:** Corriendo en **Streamlit Cloud**.
    """)

# --- TAB 1: NUEVOS ---
with tab1:
    st.header("Análisis de Nuevos Clientes")
    st.markdown("""
    **¿Qué estamos viendo?**
    Esta pestaña analiza el comportamiento de los clientes que realizaron su **primera compra hace menos de 12 meses**.
    Son el indicador principal de la tracción comercial y la capacidad de atracción de la empresa.
    """)
    
    _, arpu_n_real, _ = backend.tabla_por_mesvida(ids_nuevos, range(0, 12), cm_arpu)
    m1, m2, m3 = st.columns(3)
    m1.metric("Nuevos/Mes", backend.fmt_int(avg_new_entries), help="Promedio de clientes nuevos que entran cada mes.")
    m2.metric("ARPU Nuevos", backend.fmt_money(arpu_n_real), help="Gasto promedio mensual de un cliente nuevo durante su primer año.")
    m3.metric("Frecuencia", f"{freq_new:.2f} tx/mes", help="Promedio de veces que compran al mes.")
    
    st.subheader("Matriz de Evolución (Nuevos)")
    st.caption("Ingresos generados mes a mes por las cohortes de clientes nuevos.")
    ids_new_viz = cohort_df[cohort_df["edad_actual_meses"].between(0, 11)].index
    t_new, _, _ = backend.tabla_por_mesvida(ids_new_viz, range(0, 12), cm_arpu)
    if not t_new.empty:
        st.dataframe(t_new.applymap(lambda x: "{:,.0f}".format(x)), use_container_width=True)

# --- TAB 2: RECURRENTES ---
with tab2:
    st.header("Análisis de Clientes Recurrentes")
    st.markdown("""
    **¿Qué estamos viendo?**
    Aquí analizamos la base consolidada: clientes con antigüedad entre **12 y 36 meses**.
    Estos clientes representan la estabilidad del negocio. Suelen tener un ticket promedio más alto y predecible.
    """)
    
    _, arpu_a_real, cli_a_real = backend.tabla_por_mesvida(ids_recurrentes, range(12, 37), cm_arpu)
    m1, m2, m3 = st.columns(3)
    m1.metric("Recurrentes Activos", backend.fmt_int(cli_a_real), help="Cantidad promedio de clientes antiguos que compran mes a mes.")
    m2.metric("ARPU Recurrentes", backend.fmt_money(arpu_a_real), help="Ticket promedio mensual de la cartera fidelizada.")
    m3.metric("Frecuencia", f"{freq_rec:.2f} tx/mes", help="Frecuencia de compra de los clientes recurrentes.")
    
    st.subheader("Matriz de Evolución (Recurrentes)")
    st.caption("Comportamiento de ingresos de la base instalada a lo largo del tiempo.")
    ids_rec_viz = cohort_df[cohort_df["edad_actual_meses"].between(12, 37)].index
    t_rec, _, _ = backend.tabla_por_mesvida(ids_rec_viz, range(12, 38), cm_arpu)
    if not t_rec.empty:
        st.dataframe(t_rec.applymap(lambda x: "{:,.0f}".format(x)), use_container_width=True)

# --- TAB 3: RETENCIÓN ---
with tab3:
    st.header("Curva de Retención")
    st.markdown("""
    **¿Para qué sirve esto?**
    Muestra el porcentaje de clientes que vuelven a comprar mes tras mes después de su primera compra.
    Una curva saludable debe "aplanarse" en algún punto; si llega a cero, significa que pierdes a todos tus clientes con el tiempo.
    """)
    
    ret_view = ret_table.copy()
    ret_view.columns = ["Mes", "Elegibles", "Activos", "Retención %"]
    ret_view["Retención %"] = ret_view["Retención %"].apply(lambda x: "{:.1f}%".format(x) if pd.notnull(x) else "")
    c1, c2 = st.columns([2, 1])
    c1.line_chart(ret_table.set_index("mes_vida")["retencion_%"])
    c2.dataframe(ret_view, use_container_width=True, height=400)

# --- TAB 4: SIMULADOR YOM ---
with tab4:
    st.header("🔮 Simulador de Impacto YOM")
    st.markdown("""
    Esta calculadora proyecta el impacto financiero de activar las **4 Palancas de Crecimiento YOM**:
    1.  **Orgánico:** Dinero retenido al evitar que clientes nuevos se vayan.
    2.  **Incremental:** Dinero nuevo traído por gestión comercial activa.
    3.  **Resurrección:** Recuperación de valor de clientes perdidos históricamente.
    4.  **Recurrentes:** Aumento de ticket en la cartera actual mediante OOS y Mix.
    """)

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
    visitas_dia = col_in_1.number_input("Visitas Vendedor/Día", value=22, help="Capacidad de visitas diarias por vendedor.")
    n_vendedores = col_in_2.number_input("N° Vendedores", value=2, help="Tamaño de la fuerza de venta.")
    tasa_dig = col_in_3.number_input("Tasa Digitalización (%)", value=15.0, help="% de gestión efectiva digital.") / 100.0
    gestiones_react = col_in_4.number_input("Gestiones p/ Reactivación", value=4.0, help="Contactos necesarios para reactivar un cliente.")
    
    col_in_5, col_in_6, col_in_7, col_in_8 = st.columns(4)
    foco_act = col_in_5.number_input("Foco en Nuevos (%)", value=10.0, help="% del tiempo dedicado a caza de nuevos clientes.") / 100.0
    delta_retencion = col_in_6.number_input("Aumento Lealtad/Retención (%)", value=2.5, step=0.1, help="Meta de mejora en la tasa de retención.") / 100.0
    dias_mes = 22
    
    # Inputs Resurrección
    col_res_1, col_res_2 = st.columns(2)
    tasa_descuento_res = col_res_1.number_input("Tasa de Descuento Anual / Caída (%)", value=30.0, step=1.0, help="Tasa para descontar el valor histórico de clientes recuperados.") / 100.0
    eff_resurreccion = col_res_2.number_input("Tasa Eficiencia Resurrección (Clientes)", value=5.0, step=0.5, help="% de éxito mensual en recuperar clientes perdidos.") / 100.0

    # Inputs ARPU Recurrentes (OOS + Mix)
    col_mix_1, col_mix_2 = st.columns(2)
    oos_pct = col_mix_1.number_input("% Aumento Ingresos por OOS", value=5.0, step=0.5, help="Venta recuperada al evitar quiebres de stock.") / 100.0
    mix_pct = col_mix_2.number_input("% Aumento Ingresos por Mix", value=5.0, step=0.5, help="Venta adicional por ofrecer más variedad de productos.") / 100.0
    
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
    val_m = round(val_k * val_l)
    val_n = round(val_k * (1 - val_l))
    val_o = freq_compra_val
    val_p = val_a / val_o if val_o > 0 else 0

    with st.expander("📋 Ver Tabla de Escenario (Capacidades Calculadas)", expanded=False):
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

    # ==============================================================================
    # 0. CÁLCULOS PREVIOS (MOTOR DE SIMULACIÓN)
    # ==============================================================================
    
    # --- ORGÁNICOS ---
    clientes_extra_cohorte = np.ceil(val_b * delta_retencion)
    meses_proyeccion = 12
    matriz_org_impacto = np.zeros((meses_proyeccion, meses_proyeccion))
    matriz_org_clientes = np.zeros((meses_proyeccion, meses_proyeccion))
    for i in range(meses_proyeccion):
        for mes_vida in range(meses_proyeccion):
            idx_calendario = i + mes_vida
            if idx_calendario < meses_proyeccion:
                if mes_vida > 0:
                    matriz_org_impacto[i, idx_calendario] = clientes_extra_cohorte * val_a
                    matriz_org_clientes[i, idx_calendario] = clientes_extra_cohorte
    total_org_anual = matriz_org_impacto.sum()
    total_org_cli_12 = matriz_org_clientes[:, 11].sum()
    monthly_org_impact = total_org_anual / 12

# ==============================================================================
    # MOTOR DE SIMULACIÓN (CONECTADO A DATA REAL)
    # ==============================================================================
    
    # 1. Extraemos la retención real del gráfico (backend)
    try:
        curva_retencion_real = (ret_table.sort_values("mes_vida")["retencion_%"] / 100).tolist()
    except Exception:
        # Fallback si falla la tabla
        curva_retencion_real = [1.0, 0.60, 0.55, 0.50, 0.45, 0.40, 0.40, 0.40, 0.38, 0.38, 0.38, 0.38]

    # Aseguramos los 12 meses
    while len(curva_retencion_real) < meses_proyeccion:
        curva_retencion_real.append(curva_retencion_real[-1] if curva_retencion_real else 0.40)

    # 2. CÁLCULO DE INCREMENTALES (COHORTES)
    matriz_inc_impacto = np.zeros((meses_proyeccion, meses_proyeccion))
    matriz_inc_clientes = np.zeros((meses_proyeccion, meses_proyeccion))
    
    for i in range(meses_proyeccion):
        for mes_vida in range(meses_proyeccion):
            idx_calendario = i + mes_vida
            if idx_calendario < meses_proyeccion:
                tasa_ret_base = curva_retencion_real[mes_vida]
                if mes_vida == 0:
                    clientes_inc = val_m
                else:
                    tasa_ret_nueva = tasa_ret_base + delta_retencion
                    clientes_inc = np.round(val_m * tasa_ret_nueva)
                
                matriz_inc_impacto[i, idx_calendario] = clientes_inc * val_a
                matriz_inc_clientes[i, idx_calendario] = clientes_inc

    total_inc_anual = matriz_inc_impacto.sum()
    monthly_inc_impact = total_inc_anual / 12
    total_foco_act = val_m * 12

    # 3. CÁLCULO DE RESURRECCIÓN (NUEVA LÓGICA CON DATA REAL)
    # Definimos las variables que te daban error
    clientes_resucitados_mes = np.round(val_n * eff_resurreccion)
    acumulado_clientes_res = clientes_resucitados_mes * 12
    
    # El dinero de resurrección considera el valor histórico descontado
    # y se comporta como una cohorte que ya está en su etapa de madurez (mes 12)
    tasa_ret_res = curva_retencion_real[11] + delta_retencion
    ingreso_res_mensual = (clientes_resucitados_mes * val_a) * (1 - tasa_descuento_res)
    
    # Proyectamos el flujo de resurrección a 12 meses para obtener el impacto mensual
    matriz_res_impacto = np.zeros((meses_proyeccion, meses_proyeccion))
    for i in range(meses_proyeccion):
        for j in range(i, meses_proyeccion):
            matriz_res_impacto[i, j] = ingreso_res_mensual
            
    monthly_res_impact = matriz_res_impacto.sum() / 12
    acumulado_dinero_res = matriz_res_impacto.sum() # Para el KPI anual

    # --- RESURRECCIÓN ---
    # Dinero
    base_dinero_res = total_org_anual        
    acumulado_dinero_res = 0
    years_back = 8
    for t in range(1, years_back + 1):
        acumulado_dinero_res += base_dinero_res * ((1 - tasa_descuento_res) ** t)
    monthly_res_impact = acumulado_dinero_res / 12
    
    # Clientes
    base_clientes_res = total_org_cli_12
    acumulado_clientes_res = 0
    for t in range(1, years_back + 1):
        acumulado_clientes_res += np.round(base_clientes_res * ((1 - tasa_descuento_res) ** t))

    # --- RECURRENTES ---
    base_rec_actual = cli_a_real if pd.notnull(cli_a_real) else 0
    arpu_rec_actual = arpu_a_real if pd.notnull(arpu_a_real) else 0
    
    base_rec_future = np.ceil(base_rec_actual * (1 + delta_retencion))
    delta_clientes_rec = base_rec_future - base_rec_actual
    
    total_lift_arpu = oos_pct + mix_pct
    arpu_rec_future = arpu_rec_actual * (1 + total_lift_arpu)
    delta_arpu_rec = arpu_rec_future - arpu_rec_actual
    
    rev_monthly_actual = base_rec_actual * arpu_rec_actual
    rev_monthly_future = base_rec_future * arpu_rec_future
    delta_rev_monthly_rec = rev_monthly_future - rev_monthly_actual

    # --- TOTALES FINALES ---
    total_monthly_new_impact = monthly_org_impact + monthly_inc_impact + monthly_res_impact
    total_monthly_grand_impact = total_monthly_new_impact + delta_rev_monthly_rec

    # ==============================================================================
    # 2. TABLA RESUMEN EJECUTIVA
    # ==============================================================================
    st.subheader("📊 Resumen Ejecutivo de Impacto YOM")
    st.info("Este cuadro consolida todos los beneficios financieros estimados por las iniciativas comerciales.")
    
    st.markdown("#### A. Impacto en Nuevos Clientes (Adquisición & Recuperación)")
    df_resumen_nuevos = pd.DataFrame([
        {"Concepto": "ARPU Nuevos", "Valor": fmt_clp(val_a)},
        {"Concepto": "+ Orgánicos (Clientes Ganados)", "Valor": f"{int(total_org_cli_12)}"},
        {"Concepto": "+ Foco Activación (Clientes Ganados)", "Valor": f"{int(total_foco_act)}"},
        {"Concepto": "+ Leales (Resurrección Clientes)", "Valor": f"{int(acumulado_clientes_res)}"},
        {"Concepto": "Ingreso Adicional Mensual (Total Nuevos)", "Valor": fmt_clp(total_monthly_new_impact)},
    ])
    st.dataframe(df_resumen_nuevos, use_container_width=True)
    
    st.markdown("#### B. Impacto en Clientes Recurrentes (Cartera Actual)")
    df_resumen_rec = pd.DataFrame([
        {"Concepto": "ARPU Recurrentes (Base)", "Valor": fmt_clp(arpu_rec_actual)},
        {"Concepto": "+ Cartera Activa (Clientes Retenidos)", "Valor": f"{int(delta_clientes_rec)}"},
        {"Concepto": "+ Aumento Ganancias/Cliente (OOS+Mix)", "Valor": fmt_clp(delta_arpu_rec)},
        {"Concepto": "Ingreso Adicional Mensual (Recurrentes)", "Valor": fmt_clp(delta_rev_monthly_rec)},
    ])
    st.dataframe(df_resumen_rec, use_container_width=True)

    st.metric("💰 IMPACTO TOTAL MENSUAL (A+B)", fmt_clp(total_monthly_grand_impact))
    
    st.markdown("---")
    st.markdown("### 🔍 Desglose Detallado de Cálculos")
    st.caption("A continuación, el detalle matemático de cada una de las cifras resumidas arriba.")

    # =========================================================
    # SECCIÓN 3: ANÁLISIS ORGÁNICO
    # =========================================================
    st.subheader(f"1. Crecimiento Orgánico (Retención +{delta_retencion*100:.1f}%)")
    st.markdown(f"**Lógica:** Ingresos de clientes que *no se van* gracias a la mejora en retención, aplicados a los {val_b:.1f} nuevos clientes que entran orgánicamente cada mes.")
    
    cols_meses = list(range(1, meses_proyeccion + 1))
    idx_cohortes = [f"Cohorte {x+1}" for x in range(meses_proyeccion)]
    
    df_organico = pd.DataFrame(matriz_org_impacto, columns=cols_meses, index=idx_cohortes)
    df_organico_clientes = pd.DataFrame(matriz_org_clientes, columns=cols_meses, index=idx_cohortes)
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Impacto Orgánico Anual ($)", fmt_clp(total_org_anual))
    k2.metric("+Clientes Activos (Mes 12)", f"{int(total_org_cli_12)}")
    k3.metric("Clientes Salvados/Cohorte", f"{int(clientes_extra_cohorte)}")
    
    st.bar_chart(df_organico.sum(axis=0))
    
    with st.expander("Ver Detalle Orgánico"):
        tab_org_1, tab_org_2 = st.tabs(["💵 Ingresos ($)", "👥 Clientes (#)"])
        with tab_org_1:
            st.dataframe(df_organico.applymap(fmt_clp), use_container_width=True)
        with tab_org_2:
            st.dataframe(df_organico_clientes.applymap(lambda x: f"{int(x)}" if x > 0 else "-"), use_container_width=True)
        
    st.markdown("---")

    # =========================================================
    # SECCIÓN 4: ANÁLISIS INCREMENTALES (ACTIVACIÓN)
    # =========================================================
    st.subheader(f"2. Clientes Incrementales (Activación Comercial)")
    st.markdown(f"**Lógica:** Ingresos generados por los {int(val_m)} clientes extra mensuales captados por la fuerza de venta.")
    
    df_incremental = pd.DataFrame(matriz_inc_impacto, columns=cols_meses, index=idx_cohortes)
    df_incremental_clientes = pd.DataFrame(matriz_inc_clientes, columns=cols_meses, index=idx_cohortes)
    
    k1, k2 = st.columns(2)
    k1.metric("Impacto Incremental Anual ($)", fmt_clp(total_inc_anual))
    k2.metric("+Foco Activación (Clientes Totales)", f"{int(total_foco_act)}")
    
    st.bar_chart(df_incremental.sum(axis=0))

    with st.expander("Ver Detalle Incremental"):
        tab_inc_1, tab_inc_2 = st.tabs(["💵 Ingresos ($)", "👥 Clientes (#)"])
        with tab_inc_1:
            st.dataframe(df_incremental.applymap(fmt_clp), use_container_width=True)
        with tab_inc_2:
            st.dataframe(df_incremental_clientes.applymap(lambda x: f"{int(x)}" if x > 0 else "-"), use_container_width=True)

    st.markdown("---")

    # =========================================================
    # SECCIÓN 5: ANÁLISIS RESURRECCIÓN
    # =========================================================
    st.subheader(f"3. Clientes Resurrección (Recuperación de Cartera)")
    st.markdown("**Lógica:** Valor recuperado de clientes históricos perdidos, proyectado a 8 años.")
    st.caption(f"Descuento aplicado: {tasa_descuento_res*100:.0f}%.")
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Clientes Resucitados", f"{int(acumulado_clientes_res)}")
    k2.metric("Impacto Resurrección Anual ($)", fmt_clp(acumulado_dinero_res))
    k3.metric("Impacto Mensual ($)", fmt_clp(monthly_res_impact))
    
    with st.expander("Ver Detalle de la Proyección (8 Años)"):
        st.info("Cálculo basado en proyección histórica descontada.")

    st.markdown("---")
    
    # =========================================================
    # SECCIÓN 6: IMPACTO EN CARTERA RECURRENTES
    # =========================================================
    st.subheader("4. Impacto en Cartera de Recurrentes (OOS + Mix + Retención)")
    st.markdown("**Lógica:** Aumento del ARPU en clientes actuales mediante gestión de quiebres (OOS) y Mix.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Aumento Cartera Activa", f"+{int(delta_clientes_rec)} Clientes")
    m2.metric("Aumento ARPU (OOS+Mix)", fmt_clp(delta_arpu_rec), f"+{total_lift_arpu*100:.1f}%")
    m3.metric("Impacto Mensual Recurrentes", fmt_clp(delta_rev_monthly_rec))

    with st.expander("Ver Comparativa Antes vs Después"):
        df_compare = pd.DataFrame({
            "Métrica": ["Base Clientes Activos", "ARPU Mensual", "Ingreso Mensual Total"],
            "Situación Actual": [f"{int(base_rec_actual)}", fmt_clp(arpu_rec_actual), fmt_clp(rev_monthly_actual)],
            "Con YOM (Simulado)": [f"{int(base_rec_future)}", fmt_clp(arpu_rec_future), fmt_clp(rev_monthly_future)],
            "Diferencia (Delta)": [f"+{int(delta_clientes_rec)}", f"+{fmt_clp(delta_arpu_rec)}", f"+{fmt_clp(delta_rev_monthly_rec)}"]
        })
        st.dataframe(df_compare, use_container_width=True)

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
