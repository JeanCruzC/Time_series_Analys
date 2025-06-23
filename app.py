import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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
    df['intervalo'] = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
    df['semana_mes'] = df['fecha'].apply(lambda x: (x.day - 1) // 7 + 1)
    df['dia_semana'] = df['fecha'].dt.day_name()
    df['desvio'] = df['reales'] - df['planificados']
    df['desvio_%'] = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100

    orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=orden_dias, ordered=True)

    # ----- GRÁFICO 1: Interactivo Plotly completo -----
    st.subheader("📊 Gráfico 1: Tendencia Reales vs Planificados (Interactivo)")
    trend = df.groupby('fecha')[['planificados', 'reales']].sum().reset_index()
    trend_long = pd.melt(trend, id_vars='fecha', value_vars=['planificados', 'reales'],
                         var_name='Tipo', value_name='Volumen')

    color_map = {
        'planificados': 'orange',
        'reales': 'blue'
    }

    fig1 = px.line(
        trend_long,
        x='fecha',
        y='Volumen',
        color='Tipo',
        color_discrete_map=color_map,
        title="Contactos diarios",
        line_shape='linear'
    )

    fig1.update_traces(line=dict(width=2))

    fig1.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Volumen',
        legend_title_text='Tipo de Contacto',
        hovermode="x unified",
        dragmode="pan",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=14, label="14d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="Todo")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date",
            fixedrange=False
        ),
        yaxis=dict(fixedrange=False)
    )

    st.plotly_chart(fig1, use_container_width=True, config={"scrollZoom": True})

    # ----- GRÁFICO 2: Desvío promedio por intervalo -----
    st.subheader("📉 Gráfico 2: Desvío Promedio por Intervalo")
    interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    interval_avg.plot(kind='bar', ax=ax2)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_ylabel("% Desvío")
    ax2.set_title("Promedio de Desvío % por Intervalo")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # ----- GRÁFICO 3: Heatmap día-intervalo -----
    st.subheader("🔥 Gráfico 3: Heatmap Día - Intervalo")
    heatmap_data = df.pivot_table(index='intervalo', columns='dia_semana', values='desvio_%', aggfunc='mean')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0, ax=ax3)
    ax3.set_title("Heatmap % Desvío por Intervalo y Día de la Semana")
    st.pyplot(fig3)

    # ----- PROYECCIÓN DE AJUSTES -----
    st.subheader("📆 Proyección Semana 23/06 al 29/06")
    ajustes = df.groupby(['dia_semana', 'intervalo'])['desvio_%'].mean().reset_index()
    ajustes['ajuste_sugerido'] = (ajustes['desvio_%'].round(2)) / 100
    ajustes['semana_objetivo'] = "2025-06-23 al 2025-06-29"
    ajustes = ajustes[['semana_objetivo', 'dia_semana', 'intervalo', 'ajuste_sugerido']]

    st.dataframe(ajustes, use_container_width=True)

    st.download_button(
        "📥 Descargar ajustes proyectados (.csv)",
        data=ajustes.to_csv(index=False),
        file_name="ajustes_proyectados_2306.csv",
        mime='text/csv'
    )
