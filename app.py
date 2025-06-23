import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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
df['fecha'] = pd.to_datetime(df['fecha'])
df['intervalo'] = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week     # semana ISO (lunes–domingo)
df['mes']       = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio']    = df['reales'] - df['planificados']
df['desvio_%']  = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100

# fijar orden de días
dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# 3. Selector de vista interactiva
st.subheader("🔎 Vista interactiva: día / semana / mes")
vista = st.selectbox("Ver por:", ["Día", "Semana", "Mes"])

if vista == "Día":
    ag = df.groupby(['fecha','dia_semana'])[['planificados','reales']].sum().reset_index()
    x     = 'fecha'
    title = "📅 Contactos diarios"
elif vista == "Semana":
    ag = ( df.groupby(['semana_iso','nombre_mes'])[['planificados','reales']]
             .sum()
             .reset_index()
         )
    ag['etiqueta'] = ag['nombre_mes'] + " – Sem " + ag['semana_iso'].astype(str)
    x     = 'etiqueta'
    title = "📆 Contactos por semana ISO"
else:  # Mes
    ag = df.groupby(['mes','nombre_mes'])[['planificados','reales']].sum().reset_index()
    ag['etiqueta'] = ag['nombre_mes']
    x     = 'etiqueta'
    title = "📊 Contactos por mes"

# dar formato long para plotly
long = ag.melt(id_vars=[x], value_vars=['planificados','reales'],
               var_name='Tipo', value_name='Volumen')

color_map = {'planificados':'orange','reales':'blue'}

fig = px.line(
    long, x=x, y='Volumen', color='Tipo',
    color_discrete_map=color_map,
    title=title,
    line_shape='linear'
)
fig.update_traces(line=dict(width=2))

# si es por día, ponemos slider y selector
if vista == "Día":
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=7,  label="7d", step="day",   stepmode="backward"),
                dict(count=14, label="14d",step="day",   stepmode="backward"),
                dict(count=1,  label="1m", step="month", stepmode="backward"),
                dict(step="all", label="Todo")
            ]),
            rangeslider=dict(visible=True),
            type="date",
            fixedrange=False
        )
    )

fig.update_layout(
    hovermode="x unified",
    dragmode="pan",
    yaxis=dict(fixedrange=False)
)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


# 4. Análisis adicional — mantenido igual
st.subheader("📉 Gráfico: Desvío Promedio por Intervalo")
interval_avg = df.groupby('intervalo')['desvio_%'].mean().sort_index()
fig2, ax2 = plt.subplots(figsize=(12,4))
interval_avg.plot(kind='bar', ax=ax2)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel("% Desvío")
ax2.set_title("Promedio de Desvío % por Intervalo")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("🔥 Heatmap: Desvío por Día de la Semana y Intervalo")
heat = df.pivot_table(values='desvio_%',
                     index='intervalo',
                     columns='dia_semana',
                     aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap % Desvío")
st.pyplot(fig3)

st.subheader("📆 Proyección Ajustes (Semana 23/06 - 29/06)")
aj = df.groupby(['dia_semana','intervalo'])['desvio_%']\
       .mean().reset_index()
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
