import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo – Vista Día / Semana / Mes")

#──────────────────────────────── 1. CARGA ─────────────────────────────
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

#──────────────────────────────── 2. PREPROCESO ────────────────────────
df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['hora_str']   = df['intervalo'].astype(str).str.slice(0,5)              # 00:00
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['anno']       = df['fecha'].dt.isocalendar().year
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['dt']         = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)

# lunes de la ISO-semana para construir eje continuo
df['monday'] = df['fecha'] - pd.to_timedelta(df['fecha'].dt.weekday, unit='d')
df['desvio_%'] = (df['reales'] - df['planificados']) / df['planificados'].replace(0, np.nan) * 100

dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

vista = st.selectbox("🔎 Ver por:", ["Día","Semana","Mes"])

#─────────────────────────────── 3A. VISTA DÍA ─────────────────────────
if vista == "Día":
    daily = df.groupby('dt')[['planificados','reales']].sum().reset_index()
    fig_day = px.line(
        daily, x='dt', y=['planificados','reales'],
        labels={'value':'Volumen','dt':'Fecha y Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📅 Contactos por Intervalo (Fecha + Hora)"
    )
    fig_day.update_traces(line=dict(width=2))
    fig_day.update_xaxes(
        tickformat='%Y-%m-%d<br>%H:%M',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=6,  label="6h",  step="hour", stepmode="backward"),
            dict(count=12, label="12h", step="hour", stepmode="backward"),
            dict(count=1,  label="1d",  step="day",  stepmode="backward"),
            dict(step="all", label="Todo")
        ])
    )
    fig_day.update_layout(hovermode="x unified", dragmode="zoom")
    st.plotly_chart(fig_day, use_container_width=True,
                    config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]})

#─────────────────────────────── 3B. VISTA SEMANA ──────────────────────
elif vista == "Semana":
    st.subheader("📆 Curva horaria concatenada por Semana (zoom igual que Día)")

    # Agregamos por semana + intervalo
    week_curve = (df.groupby(['anno','semana_iso','monday','intervalo','hora_str'])
                    [['planificados','reales']].sum().reset_index())

    # Construimos datetime continuo: lunes ISO + hora del intervalo
    week_curve['dt_week'] = week_curve.apply(
        lambda r: r['monday'] + datetime.timedelta(
            hours=r['intervalo'].hour, minutes=r['intervalo'].minute), axis=1)

    fig_week = px.line(
        week_curve,
        x='dt_week',
        y=['planificados','reales'],
        labels={'value':'Volumen','dt_week':'Semana – Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="Serie contínua Semana – Hora (todas las semanas)"
    )
    fig_week.update_traces(line=dict(width=2))
    fig_week.update_xaxes(
        tickformat='%Y-W%V<br>%H:%M',              # ej. 2025-W18   12:00
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=1, label="1w", step="week", stepmode="backward"),
            dict(count=2, label="2w", step="week", stepmode="backward"),
            dict(step="all", label="Todo")
        ])
    )
    fig_week.update_layout(hovermode="x unified", dragmode="zoom")
    st.plotly_chart(fig_week, use_container_width=True,
                    config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]})

#─────────────────────────────── 3C. VISTA MES ─────────────────────────
else:
    st.subheader("📊 Curva horaria agregada por Mes")
    df_mes = df.groupby(['nombre_mes','hora_str'])[['planificados','reales']].sum().reset_index()
    for col, titulo in zip(['planificados','reales'], ['Planificados','Reales']):
        fig_mes = px.line(
            df_mes, x='hora_str', y=col, color='nombre_mes',
            labels={'hora_str':'Hora','nombre_mes':'Mes', col:'Volumen'},
            title=f"{titulo} – Curva Horaria por Mes"
        )
        fig_mes.update_traces(line=dict(width=2))
        fig_mes.update_xaxes(tickangle=45)
        st.plotly_chart(fig_mes, use_container_width=True)

#──────────────────────────── 4. ANÁLISIS EXTRA ────────────────────────
st.subheader("📉 Desvío Promedio por Intervalo")
interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
fig_bar, ax_bar = plt.subplots(figsize=(12,4))
interval_avg.plot(kind='bar', ax=ax_bar, color='steelblue')
ax_bar.axhline(0, color='black', linestyle='--')
ax_bar.set_ylabel("% Desvío")
ax_bar.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig_bar)

st.subheader("🔥 Heatmap: Desvío por Día de la Semana y Intervalo")
heat = df.pivot_table(values='desvio_%', index='intervalo', columns='dia_semana', aggfunc='mean')
fig_hm, ax_hm = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax_hm)
ax_hm.set_title("Heatmap % Desvío")
st.pyplot(fig_hm)

st.subheader("📆 Proyección Ajustes (Semana 23/06 – 29/06)")
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
