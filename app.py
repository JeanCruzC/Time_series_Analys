import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

# ─────────── Configuración de página ───────────
st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes por Intervalo + Vista Interactiva")

# ─────────── 1. Carga de datos ───────────
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv","xlsx"])
if not file:
    st.stop()

if file.name.endswith("xlsx"):
    df = pd.read_excel(file)
else:
    df = pd.read_csv(file)

# ─────────── 2. Preprocesamiento ───────────
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    'fecha': 'fecha',
    'tramo': 'intervalo',
    'planif. contactos': 'planificados',
    'contactos': 'reales'
})

df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes']        = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio']     = df['reales'] - df['planificados']
df['desvio_%']   = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100

# Orden de días
dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# ─────────── 3. Selector de Vista ───────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ["Día","Semana","Mes"])

# ────────────────────────────────────────────────
# 3.1 VISTA DÍA: serie continua fecha+hora
# ────────────────────────────────────────────────
if vista == "Día":
    df['dt'] = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
    ag = df.groupby('dt')[['planificados','reales']].sum().reset_index()

    fig_day = px.line(
        ag, x='dt', y=['planificados','reales'],
        labels={'value':'Volumen','dt':'Fecha y Hora','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📅 Contactos por Intervalo (Fecha + Hora)"
    )
    fig_day.update_traces(line=dict(width=2))
    fig_day.update_xaxes(
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
    fig_day.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        yaxis=dict(fixedrange=False, autorange=True)
    )
    st.plotly_chart(
        fig_day,
        use_container_width=True,
        config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]}
    )

# ────────────────────────────────────────────────
# 3.2 VISTA SEMANA: curva horaria facetada por semana
# ────────────────────────────────────────────────
elif vista == "Semana":
    weekly_detail = (
        df.groupby(['semana_iso','intervalo'])[['planificados','reales']]
          .mean()
          .reset_index()
    )
    fig_week_detail = px.line(
        weekly_detail,
        x='intervalo',
        y=['planificados','reales'],
        facet_col='semana_iso',
        facet_col_wrap=4,
        labels={
            'intervalo':'Hora', 
            'value':'Volumen', 
            'variable':'Tipo', 
            'semana_iso':'Semana ISO'
        },
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📆 Curva horaria por Semana (promedio diario)"
    )
    fig_week_detail.update_traces(line=dict(width=2))
    fig_week_detail.update_xaxes(
        tickformat='%H:%M',
        fixedrange=False
    )
    fig_week_detail.update_layout(
        showlegend=False,
        hovermode="x unified"
    )
    # Mostrar leyenda solo una vez
    fig_week_detail.for_each_trace(lambda t: t.update(showlegend=True) if t.name=='reales' else None)
    st.plotly_chart(
        fig_week_detail,
        use_container_width=True,
        config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]}
    )

# ────────────────────────────────────────────────
# 3.3 VISTA MES: curva horaria promedio diario por mes
# ────────────────────────────────────────────────
else:
    monthly_avg = (
        df.groupby(['mes','intervalo'])[['planificados','reales']]
          .mean()
          .reset_index()
    )
    monthly_avg['nombre_mes'] = monthly_avg['mes'].map(df.set_index('mes')['nombre_mes'])

    fig_mon_avg = px.line(
        monthly_avg,
        x='intervalo',
        y=['planificados','reales'],
        facet_col='nombre_mes',
        facet_col_wrap=2,
        labels={
            'intervalo':'Hora',
            'value':'Promedio diario',
            'variable':'Tipo',
            'nombre_mes':'Mes'
        },
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📊 Curva horaria promedio diario por Mes"
    )
    fig_mon_avg.update_traces(line=dict(width=2))
    fig_mon_avg.update_xaxes(tickformat='%H:%M', fixedrange=False)
    fig_mon_avg.update_layout(showlegend=False, hovermode="x unified")
    fig_mon_avg.for_each_trace(lambda t: t.update(showlegend=True) if t.name=='reales' else None)
    st.plotly_chart(fig_mon_avg, use_container_width=True,
                    config={"scrollZoom": True, "modeBarButtonsToAdd":["autoScale2d"]})

# ────────────────────────────────────────────────
# 4. Análisis adicional (manteniendo tu código)
# ────────────────────────────────────────────────
st.subheader("📉 Desvío Promedio por Intervalo")
interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
fig2, ax2 = plt.subplots(figsize=(12,4))
interval_avg.plot(kind='bar', ax=ax2, color='skyblue')
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel("% Desvío")
ax2.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("🔥 Heatmap: Desvío por Día de la Semana y Intervalo")
heat = df.pivot_table(
    values='desvio_%', index='intervalo', columns='dia_semana', aggfunc='mean'
)
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap % Desvío")
st.pyplot(fig3)

# ────────────────────────────────────────────────
# 5. Ajustes sugeridos dinámicos por semana
# ────────────────────────────────────────────────
st.subheader("📆 Ajustes sugeridos por Semana ISO")
semanas = sorted(df['semana_iso'].unique())
selected_week = st.selectbox("Selecciona Semana ISO:", semanas, index=len(semanas)-1)

df_week = df[df['semana_iso']==selected_week]
aj = (
    df_week.groupby(['dia_semana','intervalo'])['desvio_%']
           .mean()
           .reset_index()
)
aj['ajuste_sugerido'] = (aj['desvio_%']/100).round(4)
aj['Semana'] = f"Semana ISO {selected_week}"
aj = aj[['Semana','dia_semana','intervalo','ajuste_sugerido']]
st.dataframe(aj, use_container_width=True)
st.download_button(
    "📥 Descargar ajustes (.csv)",
    data=aj.to_csv(index=False),
    file_name=f"ajustes_semana_{selected_week}.csv",
    mime="text/csv"
)
