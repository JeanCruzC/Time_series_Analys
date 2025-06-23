import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo (Semana 23/06 - 29/06)")

file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv", "xlsx"])
if file:
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        'fecha': 'fecha',
        'tramo': 'intervalo',
        'planif. contactos': 'planificados',
        'contactos': 'reales'
    })

    df['fecha'] = pd.to_datetime(df['fecha'])
    df['semana_mes'] = df['fecha'].apply(lambda x: (x.day - 1) // 7 + 1)
    df['dia_semana'] = df['fecha'].dt.day_name()
    df['desvio'] = df['reales'] - df['planificados']
    df['desvio_%'] = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100

    # Orden de los días para el heatmap
    orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=orden_dias, ordered=True)

    # ----- GRÁFICO 1: Tendencia -----
    st.subheader("📊 Gráfico 1: Tendencia Reales vs Planificados")
    trend = df.groupby('fecha')[['planificados', 'reales']].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(trend['fecha'], trend['planificados'], label='Planificados', color='orange')
    ax1.plot(trend['fecha'], trend['reales'], label='Reales', color='blue')
    ax1.set_title("Contactos diarios")
    ax1.set_ylabel("Volumen")
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.legend()
    st.pyplot(fig1)

    # ----- GRÁFICO 2: Desvío por intervalo -----
    st.subheader("📉 Gráfico 2: Desvío Promedio por Intervalo")
    interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    interval_avg.plot(kind='bar', ax=ax2)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title("Promedio de Desvío % por Intervalo")
    ax2.set_ylabel("% Desvío")
    st.pyplot(fig2)

    # ----- GRÁFICO 3: Heatmap -----
    st.subheader("🔥 Gráfico 3: Heatmap Día - Intervalo")
    heatmap_data = df.pivot_table(index='intervalo', columns='dia_semana', values='desvio_%', aggfunc='mean')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0, ax=ax3)
    ax3.set_title("Heatmap % Desvío por Intervalo y Día de la Semana")
    st.pyplot(fig3)

    # ----- PROYECCIÓN -----
    st.subheader("📆 Proyección Semana 23/06 al 29/06")
    ajustes = df.groupby(['dia_semana', 'intervalo'])['desvio_%'].mean().reset_index()
    ajustes['ajuste_sugerido'] = ajustes['desvio_%'].round(2) / 100
    ajustes['semana_objetivo'] = "2025-06-23 al 2025-06-29"
    ajustes = ajustes[['semana_objetivo', 'dia_semana', 'intervalo', 'ajuste_sugerido']]

    st.dataframe(ajustes, use_container_width=True)

    # Descargar
    st.download_button(
        "📥 Descargar ajustes proyectados (.csv)",
        data=ajustes.to_csv(index=False),
        file_name="ajustes_proyectados_2306.csv",
        mime='text/csv'
    )
