import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo – Vista Día / Semana / Mes")

#──────────── 1. CARGA ────────────
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv","xlsx"])
if not file:
    st.stop()
df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={'fecha':'fecha','tramo':'intervalo',
                        'planif. contactos':'planificados','contactos':'reales'})

#──────────── 2. PREPROCESO ───────
df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['hora_str']   = df['intervalo'].astype(str).str.slice(0,5)                 # 00:00
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['anno']       = df['fecha'].dt.isocalendar().year
df['monday']     = df['fecha'] - pd.to_timedelta(df['fecha'].dt.weekday, unit='d')
df['dt']         = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio_%']   = (df['reales']-df['planificados'])/df['planificados'].replace(0,np.nan)*100

vista = st.selectbox("🔎 Ver por:", ["Día","Semana","Mes"])

#──────────── 3A. DÍA ─────────────
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
            dict(count=6,label="6h",step="hour", stepmode="backward"),
            dict(count=12,label="12h",step="hour", stepmode="backward"),
            dict(count=1,label="1d",step="day", stepmode="backward"),
            dict(step="all",label="Todo")
        ])
    )
    fig_day.update_layout(hovermode="x unified",dragmode="zoom")
    st.plotly_chart(fig_day,use_container_width=True,
                    config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

#──────────── 3B. SEMANA ──────────
elif vista == "Semana":
    st.subheader("📆 Curva horaria concatenada por Semana (zoom como Día)")

    # Agrego por semana + intervalo y reconstruyo datetime continuo
    wk = (df.groupby(['anno','semana_iso','monday','intervalo'])
            [['planificados','reales']].sum().reset_index())
    wk['dt'] = wk.apply(lambda r: r['monday'] +
                                   datetime.timedelta(hours=r['intervalo'].hour,
                                                      minutes=r['intervalo'].minute), axis=1)

    # Insertamos fila vacía (NaN) al final de cada semana para romper la línea
    segs = []
    for (y,w), g in wk.groupby(['anno','semana_iso']):
        g = g.sort_values('dt')
        segs.append(g)
        # marcador vacío 1 min después
        gap = g.tail(1).copy()
        gap['dt'] += pd.Timedelta(minutes=1)
        gap[['planificados','reales']] = np.nan
        segs.append(gap)
    wk_gap = pd.concat(segs, ignore_index=True)

    fig_wk = px.line(
        wk_gap, x='dt', y=['planificados','reales'],
        labels={'value':'Volumen','dt':'Semana – Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="Serie contínua Semana – Hora (hueco entre semanas)"
    )
    fig_wk.update_traces(line=dict(width=2))
    fig_wk.update_xaxes(
        tickformat='%G-W%V<br>%H:%M',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=7,label="1w", step="day", stepmode="backward"),
            dict(count=14,label="2w",step="day", stepmode="backward"),
            dict(step="all",label="Todo")
        ])
    )
    fig_wk.update_layout(hovermode="x unified",dragmode="zoom")
    st.plotly_chart(fig_wk,use_container_width=True,
                    config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

#──────────── 3C. MES ─────────────
else:
    st.subheader("📊 Curva horaria agregada por Mes")
    df_mes = df.groupby(['nombre_mes','hora_str'])[['planificados','reales']].sum().reset_index()
    for col,t in zip(['planificados','reales'],['Planificados','Reales']):
        fig = px.line(
            df_mes, x='hora_str', y=col, color='nombre_mes',
            labels={'hora_str':'Hora','nombre_mes':'Mes', col:'Volumen'},
            title=f"{t} – Curva Horaria por Mes"
        )
        fig.update_traces(line=dict(width=2))
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig,use_container_width=True)

#──────────── 4. DESVÍO / HEATMAP / AJUSTES ────────────
st.subheader("📉 Desvío Promedio por Intervalo")
int_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
fig_b, ax_b = plt.subplots(figsize=(12,4))
int_avg.plot(kind='bar', ax=ax_b, color='steelblue')
ax_b.axhline(0,color='black',linestyle='--')
ax_b.set_ylabel("% Desvío")
ax_b.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig_b)

st.subheader("🔥 Heatmap: Desvío por Día de la Semana y Intervalo")
heat = df.pivot_table(values='desvio_%', index='intervalo',
                      columns='dia_semana', aggfunc='mean')
fig_hm, ax_hm = plt.subplots(figsize=(10,6))
sns.heatmap(heat,cmap="coolwarm",center=0,ax=ax_hm)
ax_hm.set_title("Heatmap % Desvío")
st.pyplot(fig_hm)

st.subheader("📆 Proyección Ajustes (23/06 – 29/06)")
aj = df.groupby(['dia_semana','intervalo'])['desvio_%'].mean().reset_index()
aj['ajuste_sugerido'] = aj['desvio_%'].round(2)/100
aj['semana_obj'] = "2025-06-23 al 2025-06-29"
aj = aj[['semana_obj','dia_semana','intervalo','ajuste_sugerido']]
st.dataframe(aj,use_container_width=True)
st.download_button("📥 Descargar ajustes (.csv)",
                   data=aj.to_csv(index=False),
                   file_name="ajustes_2306.csv",
                   mime="text/csv")
