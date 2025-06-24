import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo + Vista Interactiva")

# ──────────────── 1. Carga de datos ────────────────
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

# ──────────────── 2. Preprocesamiento ────────────────
df['fecha'] = pd.to_datetime(df['fecha'])
df['intervalo'] = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes'] = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio'] = df['reales'] - df['planificados']
df['desvio_%'] = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100

# Asegurar orden de días para el heatmap
dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# ──────────────── 3. Vista interactiva ────────────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ["Día", "Semana", "Mes"])

# ———————————————————————————————————
# VISTA DÍA: curva contínua Fecha+Hora
# ———————————————————————————————————
if vista == "Día":
    df['dt'] = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
    ag = df.groupby('dt')[['planificados','reales']].sum().reset_index()
    fig = px.line(
        ag, x='dt', y=['planificados','reales'],
        labels={'value':'Volumen','dt':'Fecha y Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📅 Contactos por Intervalo (Fecha + Hora)"
    )
    fig.update_traces(line=dict(width=2))
    fig.update_xaxes(
        tickformat='%Y-%m-%d<br>%H:%M',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=6,  label="6h",  step="hour",  stepmode="backward"),
            dict(count=12, label="12h", step="hour",  stepmode="backward"),
            dict(count=1,  label="1d",  step="day",   stepmode="backward"),
            dict(step="all", label="Todo")
        ]),
        type="date", fixedrange=False
    )
    fig.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        yaxis=dict(fixedrange=False, autorange=True)
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]}
    )

# ———————————————————————————————————
# VISTA SEMANA: totales + curva horaria concatenada
# ———————————————————————————————————
elif vista == "Semana":
    # 3.1 Totales semanales con zoom & scroll
    df['week_start'] = df['fecha'] - pd.to_timedelta(df['fecha'].dt.weekday, unit='d')
    weekly = df.groupby('week_start')[['planificados','reales']].sum().reset_index()
    fig_wk = px.line(
        weekly, x='week_start', y=['planificados','reales'],
        labels={'value':'Volumen','week_start':'Semana (lunes)','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📆 Contactos por Semana ISO (Interactivo)"
    )
    fig_wk.update_traces(line=dict(width=2))
    fig_wk.update_xaxes(
        tickformat='%Y-%m-%d',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=7,  label="1w", step="day",   stepmode="backward"),
            dict(count=14, label="2w", step="day",   stepmode="backward"),
            dict(step="all", label="Todo")
        ]),
        type="date", fixedrange=False
    )
    fig_wk.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        yaxis=dict(fixedrange=False, autorange=True)
    )
    st.plotly_chart(
        fig_wk, use_container_width=True,
        config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]}
    )

    # 3.2 Curva horaria concatenada por semana en UNA sola serie
    df['dt_week_hour'] = df.apply(
        lambda r: datetime.datetime.combine(
            (r['fecha'] - pd.to_timedelta(r['fecha'].weekday(), unit='d')).date(),
            r['intervalo']
        ), axis=1
    )
    weekly_detail = df.groupby('dt_week_hour')[['planificados','reales']].sum().reset_index()
    fig_wk_detail = px.line(
        weekly_detail, x='dt_week_hour', y=['planificados','reales'],
        labels={'value':'Volumen','dt_week_hour':'Semana – Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="🕒 Curva horaria concatenada por Semana (00:00–23:59)"
    )
    fig_wk_detail.update_traces(line=dict(width=2))
    fig_wk_detail.update_xaxes(
        tickformat='%Y-W%V<br>%H:%M',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=7,  label="1w", step="day", stepmode="backward"),
            dict(count=14, label="2w", step="day", stepmode="backward"),
            dict(step="all", label="Todo")
        ]),
        type="date", fixedrange=False
    )
    fig_wk_detail.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        yaxis=dict(fixedrange=False, autorange=True),
        showlegend=False
    )
    # Solo una leyenda
    fig_wk_detail.for_each_trace(
        lambda t: t.update(showlegend=True) if t.name=='reales' else None
    )
    st.plotly_chart(
        fig_wk_detail, use_container_width=True,
        config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]}
    )

# ———————————————————————————————————
# VISTA MES: totales + detalle horario
# ———————————————————————————————————
else:
    # 3.3 Totales mensuales con zoom & scroll
    df['month_start'] = df['fecha'].values.astype('datetime64[M]')
    monthly = df.groupby('month_start')[['planificados','reales']].sum().reset_index()
    fig_mon = px.line(
        monthly, x='month_start', y=['planificados','reales'],
        labels={'value':'Volumen','month_start':'Mes','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📊 Contactos por Mes (Interactivo)"
    )
    fig_mon.update_traces(line=dict(width=2))
    fig_mon.update_xaxes(
        tickformat='%Y-%m',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all", label="Todo")
        ]),
        type="date", fixedrange=False
    )
    fig_mon.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        yaxis=dict(fixedrange=False, autorange=True)
    )
    st.plotly_chart(
        fig_mon, use_container_width=True,
        config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]}
    )

    # 3.4 Curva horaria concatenada por mes
    df['dt_month_hour'] = df.apply(
        lambda r: datetime.datetime.combine(r['fecha'].date(), r['intervalo']), axis=1
    )
    mon_detail = df.groupby('dt_month_hour')[['planificados','reales']].sum().reset_index()
    fig_mon_detail = px.line(
        mon_detail, x='dt_month_hour', y=['planificados','reales'],
        labels={'value':'Volumen','dt_month_hour':'Fecha y Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="🗓️ Curva horaria concatenada por Mes"
    )
    fig_mon_detail.update_traces(line=dict(width=2))
    fig_mon_detail.update_xaxes(
        tickformat='%Y-%m-%d<br>%H:%M',
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(step="all", label="Todo")
        ]),
        type="date", fixedrange=False
    )
    fig_mon_detail.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        yaxis=dict(fixedrange=False, autorange=True),
        showlegend=False
    )
    fig_mon_detail.for_each_trace(
        lambda t: t.update(showlegend=True) if t.name=='reales' else None
    )
    st.plotly_chart(
        fig_mon_detail, use_container_width=True,
        config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]}
    )

# ──────────────── 4. Análisis adicional ────────────────
st.subheader("📉 Desvío Promedio por Intervalo")
interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
fig2, ax2 = plt.subplots(figsize=(12, 4))
interval_avg.plot(kind='bar', ax=ax2)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel("% Desvío")
ax2.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("🔥 Heatmap: Desvío por Día de la Semana y Intervalo")
heat = df.pivot_table(
    values='desvio_%',
    index='intervalo',
    columns='dia_semana',
    aggfunc='mean'
)
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap % Desvío por Intervalo y Día de la Semana")
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
