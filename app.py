import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo – Día / Semana / Mes")

# ───────────────── 1. Carga de datos ─────────────────
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv","xlsx"])
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

# ───────────────── 2. Preprocesamiento ─────────────────
df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['hora_str']   = df['intervalo'].astype(str).str[:5]  # "HH:MM"
df['dt']         = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['anno']       = df['fecha'].dt.isocalendar().year
df['monday']     = df['fecha'] - pd.to_timedelta(df['fecha'].dt.weekday, unit='d')
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio_%']   = (df['reales'] - df['planificados']) / df['planificados'].replace(0, np.nan) * 100

# Para el heatmap
dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(
    df['fecha'].dt.day_name(),
    categories=dias_orden,
    ordered=True
)

# Mantener copia para análisis extra
df_main = df.copy()

# ───────────────── 3. Vista interactiva ─────────────────
vista = st.selectbox("🔎 Ver por:", ["Día","Semana","Mes"])

if vista == "Día":
    st.subheader("📅 Contactos por Intervalo (Fecha + Hora)")
    daily = df.groupby('dt')[['planificados','reales']].sum().reset_index()
    fig = px.line(
        daily, x='dt', y=['planificados','reales'],
        labels={'dt':'Fecha y Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="Día: Zoom & Scroll"
    )
    fig.update_traces(line=dict(width=2))
    fig.update_xaxes(
        tickformat='%Y-%m-%d<br>%H:%M',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=6,  label="6h",  step="hour", stepmode="backward"),
            dict(count=12, label="12h", step="hour", stepmode="backward"),
            dict(count=1,  label="1d",  step="day",  stepmode="backward"),
            dict(step="all",label="Todo")
        ])
    )
    fig.update_layout(hovermode="x unified", dragmode="zoom")
    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

elif vista == "Semana":
    st.subheader("📆 Curva horaria de una ISO‐week")

    semanas = sorted(df['semana_iso'].unique())
    sel = st.selectbox("Elige Semana ISO:", semanas)

    # Filtrar solo la semana seleccionada y agrupar por hora_str
    wk = (
        df[df['semana_iso']==sel]
        .groupby('hora_str')[['planificados','reales']]
        .sum()
        .reset_index()
    )

    fig_wk = px.line(
        wk, x='hora_str', y=['planificados','reales'],
        labels={'hora_str':'Hora del Día','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title=f"Semana {sel}: Planificados vs Reales"
    )
    fig_wk.update_traces(line=dict(width=2))
    fig_wk.update_xaxes(tickangle=45)
    fig_wk.update_layout(hovermode="x unified", dragmode="zoom")
    st.plotly_chart(fig_wk, use_container_width=True,
                    config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

else:
    st.subheader("📊 Curva horaria agregada por Mes")
    mes = df.groupby(['nombre_mes','hora_str'])[['planificados','reales']].sum().reset_index()
    for col,label in [('planificados','Planificados'),('reales','Reales')]:
        fig3 = px.line(
            mes, x='hora_str', y=col, color='nombre_mes',
            labels={'hora_str':'Hora','nombre_mes':'Mes',col:'Volumen'},
            title=f"{label} – Curva Horaria por Mes"
        )
        fig3.update_traces(line=dict(width=2))
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

# ───────────────── 4. Análisis adicional ─────────────────
st.subheader("📉 Desvío Promedio por Intervalo")
int_avg = df_main.groupby('intervalo')['desvio_%'].mean().sort_index()
fig4, ax4 = plt.subplots(figsize=(12,4))
int_avg.plot(kind='bar', ax=ax4, color='steelblue')
ax4.axhline(0, color='black', linestyle='--')
ax4.set_ylabel("% Desvío")
ax4.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig4)

st.subheader("🔥 Heatmap: Desvío por Día de la Semana y Intervalo")
# Reaseguramos la columna dia_semana
df_main['dia_semana'] = pd.Categorical(
    df_main['fecha'].dt.day_name(),
    categories=dias_orden,
    ordered=True
)
heat = df_main.pivot_table(
    values='desvio_%', index='intervalo', columns='dia_semana', aggfunc='mean'
)
fig5, ax5 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax5)
ax5.set_title("Heatmap % Desvío")
st.pyplot(fig5)

st.subheader("📆 Proyección Ajustes (23/06 – 29/06)")
aj = df_main.groupby(['dia_semana','intervalo'])['desvio_%'].mean().reset_index()
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
