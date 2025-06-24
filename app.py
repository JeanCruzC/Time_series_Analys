import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo + Vista Interactiva")

# 1. Carga de datos
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv", "xlsx"])
if not file:
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    'fecha': 'fecha',
    'tramo': 'intervalo',
    'planif. contactos': 'planificados',
    'contactos': 'reales'
})

# 2. Preprocesamiento
df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes']        = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio']     = df['reales'] - df['planificados']
df['desvio_%']   = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100

dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# 3. Selector de vista
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ["Día", "Semana", "Mes"])

if vista == "Día":
    # Combina fecha+hora para eje X
    df['dt'] = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
    ag = df.groupby('dt')[['planificados','reales']].sum().reset_index()

    fig = px.line(
        ag, x='dt', y=['planificados','reales'],
        labels={'value':'Volumen','dt':'Fecha y Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📅 Contactos por Intervalo (Fecha y Hora)",
        line_shape='linear'
    )
    fig.update_traces(line=dict(width=2))

    # Eje X con salto de línea fecha/hora, selector y slider
    fig.update_xaxes(
        title='Fecha y Hora',
        tickformat='%Y-%m-%d<br>%H:%M',
        tickangle=0,
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=6,  label="6h",  step="hour",  stepmode="backward"),
            dict(count=12, label="12h", step="hour",  stepmode="backward"),
            dict(count=1,  label="1d",  step="day",   stepmode="backward"),
            dict(step="all", label="Todo")
        ]),
        type="date",
        fixedrange=False
    )

    # **Importante**: usar dragmode="zoom" para que al box-zoom la Y se reajuste
    fig.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        yaxis=dict(fixedrange=False, autorange=True)
    )

else:
    if vista == "Semana":
        ag = (df.groupby(['semana_iso','nombre_mes'])[['planificados','reales']]
                .sum().reset_index())
        ag['etiqueta'] = ag['nombre_mes'] + " – Sem " + ag['semana_iso'].astype(str)
        x = 'etiqueta'
        title = "📆 Contactos por Semana ISO"
    else:  # Mes
        ag = df.groupby(['mes','nombre_mes'])[['planificados','reales']].sum().reset_index()
        ag['etiqueta'] = ag['nombre_mes']
        x = 'etiqueta'
        title = "📊 Contactos por Mes"

    fig = px.line(
        ag, x='etiqueta', y=['planificados','reales'],
        labels={'value':'Volumen','etiqueta':('Semana' if vista=="Semana" else 'Mes'),'variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title=title
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(hovermode="x unified", dragmode="zoom", yaxis=dict(fixedrange=False, autorange=True))
    fig.update_xaxes(tickangle=-45)

# Mostrar gráfico principal
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# 4. Análisis adicional (sin cambios)
st.subheader("📉 Desvío Promedio por Intervalo")
interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
fig2, ax2 = plt.subplots(figsize=(12,4))
interval_avg.plot(kind='bar', ax=ax2)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel("% Desvío")
ax2.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("🔥 Heatmap: Desvío por Día y Intervalo")
heat = df.pivot_table(values='desvio_%', index='intervalo', columns='dia_semana', aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap % Desvío")
st.pyplot(fig3)

st.subheader("📆 Proyección Ajustes (Semana 23/06 - 29/06)")
aj = df.groupby(['dia_semana','intervalo'])['desvio_%'].mean().reset_index()
aj['ajuste_sugerido'] = aj['desvio_%'].round(2) / 100
aj['semana_obj'] = "2025-06-23 al 2025-06-29"
aj = aj[['semana_obj','dia_semana','intervalo','ajuste_sugerido']]
st.dataframe(aj, use_container_width=True)
st.download_button(
    "📥 Descargar ajustes (.csv)",
    data=aj.to_csv(index=False),
    file_name="ajustes_2306.csv",
    mime="text/csv"
)
