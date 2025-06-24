import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo – Día / Semana / Mes")

# ─────────────── 1. Carga ───────────────
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv","xlsx"])
if not file:
    st.stop()
df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    'fecha':'fecha',
    'tramo':'intervalo',
    'planif. contactos':'planificados',
    'contactos':'reales'
})

# ─────────────── 2. Preprocesamiento ───────────────
df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['dt']         = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['hora_str']   = df['intervalo'].astype(str).str[:5]
df['desvio']     = df['reales'] - df['planificados']
df['desvio_%']   = df['desvio'] / df['planificados'].replace(0, np.nan) * 100

# Para heatmap
dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['fecha'].dt.day_name(),
                                  categories=dias_orden, ordered=True)

# guardo para el análisis final
df_main = df.copy()

# ─────────────── 3. Vista interactiva ───────────────
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
    st.subheader("📆 Contactos por Semana ISO (Interactivo)")

    # 3.1 Creamos una fecha de inicio de semana para graficar un eje continuo
    df['semana_inicio'] = df['fecha'] - pd.to_timedelta(df['fecha'].dt.weekday, unit='d')
    weekly = df.groupby('semana_inicio')[['planificados','reales']].sum().reset_index()

    # 3.2 Gráfica interactiva con rangeselector y slider
    fig_wk = px.line(
        weekly, x='semana_inicio', y=['planificados','reales'],
        labels={'semana_inicio':'Semana (lunes)','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="Semana: Totales con Zoom & Scroll"
    )
    fig_wk.update_traces(line=dict(width=2))
    fig_wk.update_xaxes(
        tickformat='%Y-%m-%d',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=2, label="2w", step="week", stepmode="backward"),
            dict(count=4, label="4w", step="week", stepmode="backward"),
            dict(step="all", label="Todo")
        ])
    )
    fig_wk.update_layout(hovermode="x unified", dragmode="zoom",
                         yaxis=dict(fixedrange=False))
    st.plotly_chart(fig_wk, use_container_width=True,
                    config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

else:
    st.subheader("📊 Contactos por Mes (Interactivo)")
    monthly = df.groupby('nombre_mes')[['planificados','reales']].sum().reset_index()
    # convertimos nombre_mes en orden cronológico
    meses_orden = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    monthly['nombre_mes'] = pd.Categorical(monthly['nombre_mes'],
                                           categories=meses_orden, ordered=True)
    fig_mon = px.line(
        monthly.sort_values('nombre_mes'),
        x='nombre_mes', y=['planificados','reales'],
        labels={'nombre_mes':'Mes','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="Mes: Totales con Zoom & Scroll"
    )
    fig_mon.update_traces(line=dict(width=2))
    fig_mon.update_xaxes(rangeslider=dict(visible=True))
    fig_mon.update_layout(hovermode="x unified", dragmode="zoom")
    st.plotly_chart(fig_mon, use_container_width=True,
                    config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

# ─────────────── 4. Análisis Adicional ───────────────
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
