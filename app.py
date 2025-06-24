import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

# --- Página ---
st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo + Vista Interactiva")

# --- 1. Carga de datos ---
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

# --- 2. Preprocesamiento ---
df['fecha'] = pd.to_datetime(df['fecha'])
df['intervalo'] = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
# Fecha + Hora para ejes
 df['dt'] = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
# Variables temporales
 df['semana_iso'] = df['fecha'].dt.isocalendar().week
 df['nombre_mes'] = df['fecha'].dt.strftime('%B')
 df['desvio'] = df['reales'] - df['planificados']
 df['desvio_%'] = df['desvio'] / df['planificados'].replace(0, np.nan) * 100

# Categorizar día de la semana
 dias = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
 df['dia_semana'] = pd.Categorical(df['fecha'].dt.day_name(), categories=dias, ordered=True)

# --- 3. Selector Día / Semana / Mes ---
vista = st.selectbox("🔎 Ver por:", ["Día", "Semana", "Mes"])

if vista == "Día":
    st.subheader("📅 Contactos por Intervalo (Fecha + Hora)")
    daily = df.groupby('dt')[['planificados','reales']].sum().reset_index()
    fig = px.line(
        daily, x='dt', y=['planificados','reales'],
        labels={'dt':'Fecha y Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="Zoom & Scroll Día"
    )
    fig.update_traces(line=dict(width=2))
    fig.update_xaxes(rangeslider=dict(visible=True), type="date")
    fig.update_layout(dragmode="pan", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

elif vista == "Semana":
    st.subheader("📆 Curvas horarias por Semana (00:00–23:59) en pequeños paneles")
    df_w = df.copy()
    df_w['hora'] = df_w['intervalo'].astype(str).str.slice(0,5)
    curve = df_w.groupby(['semana_iso','hora'])[['planificados','reales']].sum().reset_index()
    fig = px.line(
        curve, x='hora', y=['planificados','reales'],
        facet_col='semana_iso', facet_col_wrap=4,
        labels={'hora':'Hora del Día','value':'Volumen','variable':'Tipo','semana_iso':'Semana ISO'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="Curva horaria concatenada por Semana"
    )
    fig.update_traces(line=dict(width=2))
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("📊 Contactos por Mes (Totales)")
    monthly = df.groupby('nombre_mes')[['planificados','reales']].sum().reset_index()
    meses_orden = ['January','February','March','April','May','June','July',
                   'August','September','October','November','December']
    monthly['nombre_mes'] = pd.Categorical(monthly['nombre_mes'], categories=meses_orden, ordered=True)
    monthly = monthly.sort_values('nombre_mes')
    fig = px.line(
        monthly, x='nombre_mes', y=['planificados','reales'],
        labels={'nombre_mes':'Mes','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="Zoom & Scroll Mes"
    )
    fig.update_traces(line=dict(width=2))
    fig.update_xaxes(rangeslider=dict(visible=True), type="category")
    fig.update_layout(dragmode="pan", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# --- 4. Análisis extra (igual que antes) ---
st.subheader("📉 Desvío Promedio por Intervalo")
int_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
fig2, ax2 = plt.subplots(figsize=(12,4))
int_avg.plot(kind='bar', ax=ax2, color='steelblue')
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel("% Desvío")
ax2.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("🔥 Heatmap: Desvío % por Día de la Semana e Intervalo")
heat = df.pivot_table(values='desvio_%', index='intervalo', columns='dia_semana', aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap % Desvío")
st.pyplot(fig3)

st.subheader("📆 Proyección Ajustes (Semana 23/06 - 29/06)")
aj = df.groupby(['dia_semana','intervalo'])['desvio_%'].mean().reset_index()
aj['ajuste_sugerido'] = aj['desvio_%'].round(2)/100
aj['semana_obj'] = "2025-06-23 al 2025-06-29"
aj = aj[['semana_obj','dia_semana','intervalo','ajuste_sugerido']]
st.dataframe(aj, use_container_width=True)
st.download_button(
    "📥 Descargar ajustes (.csv)",
    data=aj.to_csv(index=False),
    file_name="ajustes_2306.csv",
    mime="text/csv"
)
