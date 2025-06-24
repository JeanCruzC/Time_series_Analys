import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo – Vista Día / Semana / Mes")

# 1. Carga de datos
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

# 2. Enriquecimiento
df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['nombre_mes'] = df['fecha'].dt.strftime('%B')

# 3. Selector de vista
vista = st.selectbox("🔎 Ver por:", ["Día","Semana","Mes"])

# ————————————————————————  DÍA  ————————————————————————
if vista == "Día":
    df['dt'] = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
    ag = df.groupby('dt')[['planificados','reales']].sum().reset_index()

    fig_day = px.line(
        ag, x='dt', y=['planificados','reales'],
        labels={'value':'Volumen','dt':'Fecha y Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📅 Contactos por Intervalo (Fecha + Hora)",
        line_shape='linear'
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
        ]),
        type="date"
    )
    fig_day.update_layout(hovermode="x unified", dragmode="zoom")
    st.plotly_chart(fig_day, use_container_width=True,
                    config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]})

# ——————————————————————— SEMANA ———————————————————————
elif vista == "Semana":
    sem_disponibles = sorted(df['semana_iso'].unique())
    semana_sel = st.selectbox("Elige Semana ISO:", sem_disponibles, index=len(sem_disponibles)-1)

    df_sem = df[df['semana_iso']==semana_sel]
    cur = df_sem.groupby('intervalo')[['planificados','reales']].sum().reset_index()
    cur = cur.sort_values('intervalo')
    cur['hora'] = cur['intervalo'].astype(str).str.slice(0,5)   # 00:00

    fig_sem = px.line(
        cur, x='hora', y=['planificados','reales'],
        labels={'value':'Volumen','hora':'Hora del Día','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title=f"📆 Semana ISO {semana_sel} – Curva Horaria"
    )
    fig_sem.update_traces(line=dict(width=2))
    fig_sem.update_layout(hovermode="x unified")
    fig_sem.update_xaxes(tickangle=45)
    st.plotly_chart(fig_sem, use_container_width=True)

# ————————————————————————  MES  ————————————————————————
else:   # vista == "Mes"
    meses_disp = df['nombre_mes'].unique().tolist()
    mes_sel = st.selectbox("Elige Mes:", meses_disp, index=len(meses_disp)-1)

    df_mes = df[df['nombre_mes']==mes_sel]
    cur_m = df_mes.groupby('intervalo')[['planificados','reales']].sum().reset_index()
    cur_m = cur_m.sort_values('intervalo')
    cur_m['hora'] = cur_m['intervalo'].astype(str).str.slice(0,5)

    fig_mes = px.line(
        cur_m, x='hora', y=['planificados','reales'],
        labels={'value':'Volumen','hora':'Hora del Día','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title=f"📊 Mes {mes_sel} – Curva Horaria"
    )
    fig_mes.update_traces(line=dict(width=2))
    fig_mes.update_layout(hovermode="x unified")
    fig_mes.update_xaxes(tickangle=45)
    st.plotly_chart(fig_mes, use_container_width=True)
