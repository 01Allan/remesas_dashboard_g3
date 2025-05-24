import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import json
from scipy import stats
from scipy.special import inv_boxcox
from scipy.stats import trim_mean

# Configuración de la página
st.set_page_config(
    page_title="Simulación de Remesas en Honduras",
    layout="wide",
    page_icon="🇭🇳"
)

# Sidebar con controles
st.sidebar.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOigEjEuecL2qeClB_r1HeOtpyUVMx3CA-Uw&s" width="200">
        <h4 style="text-align: center;">10386 OPTIMIZACIÓN Y SIMULACIÓN 2025Q2</h4>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Parámetros")
frecuencia = st.sidebar.selectbox("Frecuencia de análisis", ["trimestral", "anual"])
distribuciones_opciones = ["normal", "lognormal", "empirica", "weibull", "gamma", "gpd", "gmm"]
distribucion = st.sidebar.selectbox("Distribución simulada", distribuciones_opciones)
if distribucion is None:
    distribucion = distribuciones_opciones[0]
nivel_confianza = st.sidebar.slider("Nivel de confianza", 80, 99, 95)

st.sidebar.markdown("**Configuración Avanzada**")
aplicar_ventana_movil = st.sidebar.checkbox("Aplicar ventana móvil (últimos N trimestres)")
n_ventana = st.sidebar.slider("Cantidad de trimestres recientes", min_value=12, max_value=84, value=40, step=4)

st.sidebar.markdown("""<small>Datos obtenidos vía API pública del BCH</small>""", unsafe_allow_html=True)
st.sidebar.markdown("""
---
#### Integrantes del Grupo 3:
- Allan Enrique Pineda García  
- Juan Manuel Flores Zelaya
- Katherine Mabel Fiallos Antunez
- Lorna Nazareth Moncada Mendez
- Rony Filander Lainez Pacheco
---
""")
# Encabezado
st.markdown("### Dashboard — Simulación de Remesas Familiares")
st.write("Este dashboard utiliza simulaciones Monte Carlo sobre datos reales del BCH para visualizar posibles trayectorias futuras y analizar riesgo e incertidumbre.")

# BASE_URL = "http://127.0.0.1:8000/simulacion/montecarlo"
BASE_URL = "https://bff-remesas.onrender.com/simulacion/montecarlo"
url = f"{BASE_URL}/{frecuencia}/{distribucion}"

# Inicializar proyecciones_df para evitar errores de variable no definida
proyecciones_df = pd.DataFrame()
# Inicializar simulaciones_boxcox para evitar errores de variable no definida
simulaciones_boxcox = None

lambda_boxcox = None  # Inicialización preventiva
if st.button("Ejecutar Simulación"):
    with st.spinner("Consultando API y generando simulación..."):
        try:
            params = {
                "ventana_movil": "true" if aplicar_ventana_movil else "false",
                "n_ventana": n_ventana
            }
            response = requests.get(url, params=params)
            # st.write("DEBUG KEYS:", list(response.json().keys()))
            response.raise_for_status()
            data = response.json()
            if "simulaciones" not in data or not isinstance(data["simulaciones"], list) or len(data["simulaciones"]) == 0:
                st.error("La respuesta del backend no contiene simulaciones válidas.")
                st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar con la API del backend: {e}")
            st.stop()
        except ValueError:
            st.error("Error al procesar la respuesta del backend.")
            st.stop()

    if "historico" not in data:
        st.error("No se pudo cargar la simulación: faltan datos históricos en la respuesta del API.")
        st.stop()

    historico_df = pd.DataFrame(data["historico"])
    historico_df["fecha"] = pd.to_datetime(historico_df["fecha"])

    proyecciones_df = pd.DataFrame(data["proyecciones"])
    proyecciones_df["fecha"] = pd.to_datetime(proyecciones_df["fecha"])

    simulaciones = pd.Series(data.get("simulaciones", []))
    media_simulada = data.get('parametros', {}).get('media', np.mean(simulaciones) if len(simulaciones) > 0 else 0)
    std_simulada = data.get('parametros', {}).get('std', np.std(simulaciones) if len(simulaciones) > 0 else 0)

    # Estadísticas adicionales: media robusta y mediana
    media_robusta = trim_mean(simulaciones, 0.1)
    mediana_simulada = np.median(simulaciones)

    st.subheader("Estadísticas de Simulación")
    col1, col2, col3 = st.columns(3)
    col1.metric("Media clásica", f"{media_simulada:,.2f}")
    col2.metric("Media robusta (trimmed)", f"{media_robusta:,.2f}")
    col3.metric("Mediana simulada", f"{mediana_simulada:,.2f}")

    # Transformación Box-Cox y simulación con datos transformados (si todos positivos)
    simulaciones_boxcox = None
    lambda_boxcox = None
    # Comprobar que todos los valores son estrictamente positivos y la desviación estándar es mayor que cero
    try:
        if (simulaciones <= 0).any() or np.std(simulaciones) == 0:
            raise ValueError("Box-Cox requires all values to be positive and not constant.")
        boxcox_vals, lambda_boxcox = stats.boxcox(simulaciones) # type: ignore
        if abs(lambda_boxcox - 1.0) > 0.1:
            mean_bc = np.mean(boxcox_vals) # type: ignore
            std_bc = np.std(boxcox_vals)   # type: ignore
            simulaciones_boxcox = stats.norm.rvs(loc=mean_bc, scale=std_bc, size=len(simulaciones))
            simulaciones_boxcox = inv_boxcox(simulaciones_boxcox, lambda_boxcox)
            simulaciones = pd.Series(simulaciones_boxcox)
    except Exception as e:
        st.warning(f"Error en transformación Box-Cox: {e}")
    # --- Box-Cox lambda improvement for normality ---
    if lambda_boxcox is not None and isinstance(lambda_boxcox, (float, int)):
        if lambda_boxcox < 0.1:
            # Re-transform data to improve normality
            simulaciones_boxcox = stats.boxcox(simulaciones)[0]
            simulaciones = pd.Series(inv_boxcox(simulaciones_boxcox, lambda_boxcox))

    # Calcular intervalo de confianza dinámico
    percentil_inf = (100 - nivel_confianza) / 2
    percentil_sup = 100 - percentil_inf
    ic_min = np.percentile(simulaciones, percentil_inf) if len(simulaciones) > 0 else 0
    ic_max = np.percentile(simulaciones, percentil_sup) if len(simulaciones) > 0 else 0

    # Sección 1: Indicadores Clave
    st.subheader("Indicadores Clave de Datos Reales (Millones de USD)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Media Histórica", f"{historico_df['valor'].mean():,.2f}")
    col2.metric("Mínimo Histórico", f"{historico_df['valor'].min():,.2f}")
    col3.metric("Máximo Histórico", f"{historico_df['valor'].max():,.2f}")
    col4.metric(f"N° de {'Trimestres' if frecuencia=='trimestral' else 'Años'}", len(historico_df))

    # Sección 2: Análisis de Simulación
    st.subheader(f"Análisis de Simulación Monte Carlo — Distribución Simulada: :blue[{distribucion.capitalize()}]") # type: ignore

    st.markdown(f"#### Densidad de simulaciones vs distribución :blue[{distribucion.capitalize()}]")

    # Calcular KDE para histórico y simulaciones usando seaborn
    hist_kde = sns.kdeplot(x=historico_df['valor'], bw_adjust=0.5)
    x_hist = hist_kde.get_lines()[0].get_xdata()
    y_hist = hist_kde.get_lines()[0].get_ydata()
    plt.clf()
    sim_kde = sns.kdeplot(x=simulaciones.values, bw_adjust=0.5)
    sim_lines = sim_kde.get_lines()
    if len(sim_lines) > 0:
        x_sim = sim_lines[0].get_xdata()
        y_sim = sim_lines[0].get_ydata()
    else:
        x_sim = []
        y_sim = []
    plt.clf()

    fig_density_kde = go.Figure()
    fig_density_kde.add_trace(go.Scatter(x=x_hist, y=y_hist, mode="lines", name="Histórico KDE", line=dict(color="blue")))
    fig_density_kde.add_trace(go.Scatter(x=x_sim, y=y_sim, mode="lines", name="Simulado KDE", line=dict(color="purple")))
    fig_density_kde.update_layout(xaxis_title="Valor", yaxis_title="Densidad")
    st.plotly_chart(fig_density_kde, use_container_width=True)

    if lambda_boxcox is not None and isinstance(lambda_boxcox, (float, int)):
        with st.expander("¿Qué significa Lambda Box-Cox?"):
            st.markdown(f"""
            La transformación Box-Cox permite normalizar distribuciones. El parámetro \\( \\lambda = {lambda_boxcox:.4f} \\) tiene el siguiente significado:

            - $ \\lambda \\approx 1 $: No se necesita transformación.
            - $ \\lambda \\approx 0 $: Aplicar logaritmo mejora la normalidad.
            - $ \\lambda = 0 $: Datos altamente sesgados, posible transformación recíproca.
            - $ \\lambda = 1 $: Datos comprimidos a la izquierda.

            En este caso, indica que {'no se requiere transformación adicional' if abs(lambda_boxcox - 1) < 0.1 else 'se recomienda una transformación logarítmica'}.
            """, unsafe_allow_html=True)

    # Sección: Evaluación de Ajuste de Distribuciones
    ajuste = data.get("ajuste_distribuciones")
    # Asegurarse de que lambda_boxcox se obtiene del backend si está presente en ajuste_distribuciones
    if "lambda_boxcox" not in locals() or lambda_boxcox is None:
        lambda_boxcox = data.get("ajuste_distribuciones", {}).get("boxcox_lambda", None)
    if ajuste and isinstance(ajuste, dict) and len(ajuste) > 0:
        st.subheader("Comparativa de Ajuste de Distribuciones")

        distribuciones_resumen = []

        for nombre, valores in ajuste.items():
            if isinstance(valores, dict):
                fila = {
                    "Distribución": nombre.capitalize(),
                    "KS p-value": f"{valores.get('ks_pvalue', 'N/A'):.4f}" if valores.get("ks_pvalue") is not None else "N/A",
                    "Anderson-Darling": f"{valores.get('anderson_darling', 'N/A'):.4f}" if valores.get("anderson_darling") is not None else "N/A",
                    "BIC": f"{valores.get('bic', 'N/A'):.2f}" if valores.get("bic") is not None else "N/A",
                    "Componentes GMM": valores.get("n_components", "N/A")
                }
                distribuciones_resumen.append(fila)
            elif nombre == "boxcox_lambda":
                st.markdown(
                    f"**Lambda Box-Cox óptimo:** "
                    f"`{float(valores):.4f}`" if isinstance(valores, (float, int, np.float64)) else f"**Lambda Box-Cox óptimo:** `{valores}`" # type: ignore
                )

        st.markdown("**Nota**: Algunas métricas como Anderson-Darling o BIC no se aplican a todas las distribuciones, como GMM o Empírica. Por eso pueden aparecer como 'N/A'.")

        df_resumen = pd.DataFrame(distribuciones_resumen)
        st.dataframe(
            df_resumen.style
                .format({
                    "KS p-value": lambda x: f"{float(x):.4f}" if isinstance(x, (int, float)) else x,
                    "Anderson-Darling": lambda x: f"{float(x):.4f}" if isinstance(x, (int, float)) else x,
                    "BIC": lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x,
                }) # type: ignore
                .set_properties(**{"text-align": "center"})  # type: ignore
        )
        
    # --- Continuar solo si proyecciones_df no está vacío y tiene columnas válidas ---
    if proyecciones_df is not None and not proyecciones_df.empty and len(proyecciones_df.columns) > 0:

        # Trayectorias más probables: visualizar 5 líneas distintas con estilo visible
        trayectorias_mas_probables = data.get("trayectorias_mas_probables", None)
        # Trayectoria más probable: línea roja continua
        trayectoria_mas_probable = data.get("trayectoria_mas_probable", [])

        # --- Proyecciones Futuras y Visualización de Trayectorias ---
        st.markdown("#### Proyecciones Futuras (trayectorias posibles + banda IC)")
        st.markdown(
            "<b style='color: red;'>Trayectoria más probable:</b> La línea roja representa el promedio por período de todas las trayectorias simuladas, proporcionando una estimación central robusta.",
            unsafe_allow_html=True
        )
        fig_proy = go.Figure()
        fig_proy.add_trace(go.Scatter(x=historico_df["fecha"], y=historico_df["valor"],
                                    mode="lines+markers", name="Histórico", line=dict(color="#1f77b4")))

        # Trayectorias más probables (5 líneas con estilo visible)
        if isinstance(trayectorias_mas_probables, list) and len(trayectorias_mas_probables) >= 5:
            trayectorias_array = np.array(trayectorias_mas_probables)
            for i in range(5):
                fig_proy.add_trace(go.Scatter(
                    x=proyecciones_df["fecha"],
                    y=trayectorias_array[i],
                    mode="lines",
                    name=f"Trayectoria más probable {i+1}",
                    line=dict(width=2, dash="dash"),
                    opacity=0.9,
                    showlegend=True
                ))

        # Trayectoria más probable (línea roja continua)
        if trayectoria_mas_probable and len(trayectoria_mas_probable) == len(proyecciones_df):
            trayectoria_mas_probable_np = np.array(trayectoria_mas_probable)
        else:
            st.warning("No se recibió trayectoria más probable, usando media por período como aproximación.")
            trayectoria_mas_probable_np = proyecciones_df["valor"].rolling(window=2).mean().fillna(method='bfill').values # type: ignore
        fig_proy.add_trace(go.Scatter(
            x=proyecciones_df["fecha"],
            y=trayectoria_mas_probable_np,
            mode="lines+markers",
            name="Trayectoria Más Probable",
            line=dict(color="red", width=3)
        ))
        promedio_historico = historico_df["valor"].mean()
        diff_probable = np.mean(np.abs(trayectoria_mas_probable_np - promedio_historico))
        st.metric(
            label="Diferencia promedio (absoluta) entre trayectoria más probable y media histórica",
            value=f"{diff_probable:,.2f} millones USD"
        )

        # Trayectorias simuladas: usar trayectorias reales si están disponibles
        if isinstance(data.get("trayectorias"), list) and len(data["trayectorias"]) >= 20:
            trayectorias_array = np.array(data["trayectorias"])
            for i in range(20):
                fig_proy.add_trace(go.Scatter(x=proyecciones_df["fecha"], y=trayectorias_array[i], mode="lines", line=dict(width=1), name=f"Trayectoria {i}", opacity=0.2, showlegend=False))
        else:
            for i in range(20):
                valores_simulados = np.random.normal(media_simulada, std_simulada, len(proyecciones_df))
                fig_proy.add_trace(go.Scatter(x=proyecciones_df["fecha"], y=valores_simulados, mode="lines", line=dict(width=1), name=f"Trayectoria {i}", opacity=0.2, showlegend=False))
        fig_proy.add_trace(go.Scatter(
            x=proyecciones_df["fecha"].tolist() + proyecciones_df["fecha"][::-1].tolist(),
            y=[ic_max] * len(proyecciones_df) + [ic_min] * len(proyecciones_df),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name=f"IC{nivel_confianza}"
        ))
        fig_proy.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Millones de USD")
        st.plotly_chart(fig_proy, use_container_width=True)

        # Simulación vs histórico (media, bandas IC y trayectoria simulada)
        st.markdown("#### Simulación vs histórico (media, bandas IC y trayectoria simulada)")
        fig_comparativa = go.Figure()
        # Línea histórica
        fig_comparativa.add_trace(go.Scatter(
            x=historico_df["fecha"], y=historico_df["valor"],
            mode="lines+markers", name="Histórico", line=dict(color="#1f77b4")
        ))
        # Trayectoria simulada promedio (una corrida)
        trayectoria_simulada = np.random.normal(media_simulada, std_simulada, len(historico_df))
        fig_comparativa.add_trace(go.Scatter(
            x=historico_df["fecha"], y=trayectoria_simulada,
            mode="lines", name="Trayectoria Simulada", line=dict(color="orange", dash="dot")
        ))
        # Media simulada como línea horizontal
        fig_comparativa.add_trace(go.Scatter(
            x=historico_df["fecha"], y=[media_simulada]*len(historico_df),
            mode="lines", name="Media Simulada", line=dict(color="green", dash="dot")
        ))
        # Bandas de IC
        fig_comparativa.add_trace(go.Scatter(
            x=historico_df["fecha"], y=[ic_min]*len(historico_df),
            mode="lines", name=f"IC{nivel_confianza} Inferior", line=dict(color="red", dash="dot")
        ))
        fig_comparativa.add_trace(go.Scatter(
            x=historico_df["fecha"], y=[ic_max]*len(historico_df),
            mode="lines", name=f"IC{nivel_confianza} Superior", line=dict(color="red", dash="dot")
        ))
        # Promedio histórico
        promedio_historico = historico_df["valor"].mean()
        diferencia = abs(media_simulada - promedio_historico)
        fig_comparativa.add_trace(go.Scatter(
            x=historico_df["fecha"], y=[promedio_historico]*len(historico_df),
            mode="lines", name="Promedio Histórico", line=dict(color="blue", dash="dash")
        ))
        # Anotación de diferencia
        fig_comparativa.add_annotation(
            x=historico_df["fecha"].iloc[len(historico_df)//2],
            y=max(media_simulada, promedio_historico),
            text=f"Diferencia: {diferencia:,.2f} M USD",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40
        )
        fig_comparativa.update_layout(
            template="plotly_white",
            xaxis_title="Fecha",
            yaxis_title="Millones de USD"
        )
        st.plotly_chart(fig_comparativa, use_container_width=True)

        # Interpretación gerencial
        st.subheader("Interpretación Gerencial")
        riesgo_extremo = False
        mensaje = ""
        if ic_max > historico_df['valor'].max():
            mensaje += f"- El valor máximo proyectado ({ic_max:,.2f}) **supera el máximo histórico** ({historico_df['valor'].max():,.2f}).\n"
            riesgo_extremo = True
        if ic_min < historico_df['valor'].min():
            mensaje += f"-  El valor mínimo proyectado ({ic_min:,.2f}) **es menor al mínimo histórico** ({historico_df['valor'].min():,.2f}).\n"
            riesgo_extremo = True
        mensaje += f"- Diferencia absoluta entre promedio simulado y promedio histórico: {diferencia:,.2f} millones de USD.\n"
        mensaje += f"- Nivel de confianza seleccionado: IC{nivel_confianza}."
        if riesgo_extremo:
            st.error(f"""**Riesgo de Escenarios Extremos Detectado**
                La simulación proyecta valores fuera del rango histórico observado, lo cual **podría indicar alta volatilidad o incertidumbre futura**.  
                Se recomienda:
                - Reforzar monitoreo en trimestres críticos.
                - Evaluar políticas de contingencia.
                - Verificar si una distribución alternativa mejora el ajuste.

            {mensaje}
            """)
        else:
            st.success(f"""**Simulación Coherente con la Realidad**
                Los valores simulados se mantienen dentro del rango histórico. Esto indica que la distribución seleccionada **modela adecuadamente el comportamiento real** y es útil para proyecciones.

                {mensaje}
            """)

        # Sección 3: Proyecciones Futuras
        st.subheader("Proyecciones Futuras")
        st.dataframe(proyecciones_df.rename(columns={"fecha": "Fecha Proyectada", "valor": "Remesas Estimadas"}).style.set_properties(**{"text-align": "center"})) # type: ignore

        # Sección 4: Descarga de Datos
        st.subheader("Descarga de Datos")
        st.download_button("Descargar JSON (simulación)", data=json.dumps(data, indent=2).encode('utf-8'), file_name="simulacion.json", mime="application/json")
        simulaciones_df = pd.DataFrame({"simulacion": simulaciones})
        st.download_button("Descargar CSV Histórico", data=historico_df.to_csv(index=False).encode(), file_name="historico.csv")
        st.download_button("Descargar CSV Proyecciones", data=proyecciones_df.to_csv(index=False).encode(), file_name="proyecciones.csv")

    # --- Si hay simulaciones_boxcox, mostrar comparación adicional ---
    if simulaciones_boxcox is not None:
        # Asegurarse de que lambda_boxcox está definido
        if 'lambda_boxcox' not in locals() or lambda_boxcox is None: # type: ignore
            lambda_boxcox = 1.0  # Valor por defecto si no se calculó

        st.markdown("#### Comparación de densidad: Simulación Original vs Simulación Box-Cox")

        # Asegurarse de que simulaciones_boxcox es un array 1D
        if isinstance(simulaciones_boxcox, tuple):
            simulaciones_boxcox_plot = np.asarray(simulaciones_boxcox[0]).flatten()
        else:
            simulaciones_boxcox_plot = np.asarray(simulaciones_boxcox).flatten()

        sim_kde_bc = sns.kdeplot(x=simulaciones_boxcox_plot, bw_adjust=0.5)
        x_bc = sim_kde_bc.get_lines()[0].get_xdata()
        y_bc = sim_kde_bc.get_lines()[0].get_ydata()
        plt.clf()

        fig_bc = go.Figure()
        fig_bc.add_trace(go.Scatter(x=x_sim, y=y_sim, mode="lines", name="Simulación Original", line=dict(color="purple")))# type: ignore
        fig_bc.add_trace(go.Scatter(x=x_bc, y=y_bc, mode="lines", name="Simulación Box-Cox", line=dict(color="orange")))
        fig_bc.update_layout(
            xaxis_title="Valor",
            yaxis_title="Densidad",
            title="KDE: Original vs Box-Cox"
        )
        st.plotly_chart(fig_bc, use_container_width=True)
