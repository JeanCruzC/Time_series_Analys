import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo – Vista Día / Semana / Mes")

# ───────────────────────────── 1. CARGA ─────────────────────────────
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

# ───────────────────────────── 2. PREPROCESO ─────────────────────────
df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['hora_str']   = df['intervalo'].astype(str).str.slice(0,5)          # 00:00
df['dt']         = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes']        = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio_%']   = (df['reales'] - df['planificados']) / df['planificados'].replace(0, np.nan) * 100

dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# ───────────────────────────── 3. VISTA PRINCIPAL ────────────────────
vista = st.selectbox("🔎 Ver por:", ["Día", "Semana", "Mes"])

# ─────────────── DÍA ───────────────
if vista == "Día":
    daily = df.groupby('dt')[['planificados', 'reales']].sum().reset_index()
    fig_day = px.line(
        daily, x='dt', y=['planificados', 'reales'],
        color_discrete_map={'planificados':'orange', 'reales':'blue'},
        labels={'value':'Volumen', 'dt':'Fecha y Hora', 'variable':'Tipo'},
        title="📅 Contactos por Intervalo (Fecha + Hora)"
    )
    fig_day.update_traces(line=dict(width=2))
    fig_day.update_xaxes(
        tickformat='%Y-%m-%d<br>%H:%M',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=6, label="6h", step="hour", stepmode="backward"),
            dict(count=12, label="12h", step="hour", stepmode="backward"),
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(step="all", label="Todo")
        ])
    )
    st.plotly_chart(fig_day, use_container_width=True,
                    config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]})

# ─────────────── SEMANA (una sola gráfica estilo Excel) ───────────────
elif vista == "Semana":
    st.subheader("📆 Curva horaria concatenada por Semana")
    df_sem = df.sort_values(['semana_iso', 'hora_str']).copy()
    df_sem['tag'] = "Sem " + df_sem['semana_iso'].astype(str) + " – " + df_sem['hora_str']
    df_sem['tag'] = pd.Categorical(df_sem['tag'], categories=df_sem['tag'].unique(), ordered=True)

    fig_sem = px.line(
        df_sem, x='tag', y=['planificados', 'reales'],
        color_discrete_map={'planificados':'orange', 'reales':'blue'},
        labels={'value':'Volumen', 'tag':'Semana – Hora', 'variable':'Tipo'},
        title="Serie contínua Semana – Intervalo (todas las semanas)"
    )
    fig_sem.update_traces(line=dict(width=2))
    fig_sem.update_xaxes(tickangle=90)
    st.plotly_chart(fig_sem, use_container_width=True)

# ─────────────── MES ───────────────
else:
    st.subheader("📊 Curva horaria agregada por Mes")
    df_mes = df.groupby(['nombre_mes', 'hora_str'])[['planificados', 'reales']].sum().reset_index()

    fig_mes_plan = px.line(
        df_mes, x='hora_str', y='planificados',
        color='nombre_mes',
        labels={'hora_str':'Hora', 'planificados':'Volumen', 'nombre_mes':'Mes'},
        title="Planificados – Curva Horaria por Mes"
    )
    fig_mes_plan.update_traces(line=dict(width=2))
    fig_mes_plan.update_xaxes(tickangle=45)
    st.plotly_chart(fig_mes_plan, use_container_width=True)

    fig_mes_real = px.line(
        df_mes, x='hora_str', y='reales',
        color='nombre_mes',
        labels={'hora_str':'Hora', 'reales':'Volumen', 'nombre_mes':'Mes'},
        title="Reales – Curva Horaria por Mes"
    )
    fig_mes_real.update_traces(line=dict(width=2))
    fig_mes_real.update_xaxes(tickangle=45)
    st.plotly_chart(fig_mes_real, use_container_width=True)

# ───────────────────────────── 4. ANÁLISIS ADICIONAL ─────────────────
st.subheader("📉 Desvío Promedio por Intervalo")
interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
fig_int, ax_int = plt.subplots(figsize=(12, 4))
interval_avg.plot(kind='bar', ax=ax_int, color='steelblue')
ax_int.axhline(0, color='black', linestyle='--')
ax_int.set_ylabel("% Desvío")
ax_int.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig_int)

st.subheader("🔥 Heatmap: Desvío por Día de la Semana y Intervalo")
heat = df.pivot_table(values='desvio_%', index='intervalo', columns='dia_semana', aggfunc='mean')
fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax_hm)
ax_hm.set_title("Heatmap % Desvío")
st.pyplot(fig_hm)

st.subheader("📆 Proyección Ajustes (Semana 23/06 - 29/06)")
aj = df.groupby(['dia_semana', 'intervalo'])['desvio_%'].mean().reset_index()
aj['ajuste_sugerido'] = (aj['desvio_%'].round(2)) / 100
aj['semana_obj'] = "2025-06-23 al 2025-06-29"
aj = aj[['semana_obj', 'dia_semana', 'intervalo', 'ajuste_sugerido']]
st.dataframe(aj, use_container_width=True)
st.download_button(
    "📥 Descargar ajustes (.csv)",
    data=aj.to_csv(index=False),
    file_name="ajustes_2306.csv",
    mime="text/csv"
)
