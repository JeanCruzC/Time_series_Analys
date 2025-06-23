import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo (Proyección Semana 23/06 - 29/06)")

# Subida de archivo
file = st.file_uploader("Carga tu archivo de históricos", type=["csv", "xlsx"])
if file:
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    # Renombrar para consistencia
    df = df.rename(columns={
        'fecha': 'fecha',
        'tramo': 'intervalo',
        'planif. contactos': 'planificados',
        'contactos': 'reales'
    })

    # Conversión y columnas auxiliares
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['semana_mes'] = df['fecha'].apply(lambda x: (x.day - 1) // 7 + 1)
    df['dia_semana'] = df['fecha'].dt.day_name()
    df['desvio'] = df['reales'] - df['planificados']
    df['desvio_%'] = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100

    st.subheader("📊 Gráfico 1: Tendencia General")
    trend = df.groupby('fecha')[['planificados', 'reales']].sum()
    fig1, ax1 = plt.subplots()
    ax1.plot(trend.index, trend['planificados'], label='Planificados')
    ax1.plot(trend.index, trend['reales'], label='Reales')
    ax1.set_title("Contactos Reales vs Planificados por Día")
    ax1.set_ylabel("Volumen")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("📉 Gráfico 2: Desvío Promedio por Intervalo")
    interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    interval_avg.plot(kind='bar', ax=ax2)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title("Promedio de Desvío % por Intervalo")
    ax2.set_ylabel("% Desvío")
    st.pyplot(fig2)

    st.subheader("🔥 Gráfico 3: Heatmap Día - Semana - Intervalo")
    df['semana_nombre'] = "Semana " + df['semana_mes'].astype(str)
    heatmap_data = df.pivot_table(index='intervalo', columns=['dia_semana'], values='desvio_%', aggfunc='mean')
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0, ax=ax3)
    ax3.set_title("Heatmap % Desvío por Intervalo y Día de la Semana")
    st.pyplot(fig3)

    st.subheader("📆 Proyección Semana 23/06 al 29/06")
    # Usar comportamiento histórico para calcular ajustes
    ajustes = df.groupby(['dia_semana', 'intervalo'])['desvio_%'].mean().reset_index()
    ajustes['ajuste_sugerido_%'] = ajustes['desvio_%'].round(2)
    ajustes['semana_objetivo'] = "23/06 al 29/06"
    ajustes = ajustes[['semana_objetivo', 'dia_semana', 'intervalo', 'ajuste_sugerido_%']]

    st.dataframe(ajustes, use_container_width=True)

    st.download_button("📥 Descargar ajustes proyectados (Excel)", data=ajustes.to_csv(index=False), file_name="ajustes_proyectados_2306.csv")
