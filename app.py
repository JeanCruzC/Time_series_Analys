import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime as dt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# ─────────── Configuración de página ───────────
st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes + KPIs, Errores y Recomendaciones")

# ─────────── 1. Carga de datos ───────────
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv","xlsx"])
if not file:
    st.stop()
df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

# ─────────── 2. Preprocesamiento ───────────
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    'fecha':'fecha',
    'tramo':'intervalo',
    'planif. contactos':'planificados',
    'contactos':'reales'
})
df['fecha'] = pd.to_datetime(df['fecha'])
df['intervalo'] = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes'] = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio'] = df['reales'] - df['planificados']
df['desvio_%'] = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100
# Orden de días
dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# ─────────── 2.1 Serie continua ───────────
df['_dt'] = df.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_continua = df.groupby('_dt')[['planificados','reales']].sum().sort_index()

# ─────────── 2.2 Última semana ───────────
ultima_sem = df['semana_iso'].max()
_df_last = df[df['semana_iso']==ultima_sem].copy()
_df_last['_dt'] = _df_last.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_last = _df_last.groupby('_dt')[['planificados','reales']].sum().sort_index()

# ─────────── 3. Ajustes sugeridos con combinación ───────────
st.subheader("📆 Ajustes sugeridos")
semanas = sorted(df['semana_iso'].unique())
sem_sel = st.selectbox("Selecciona Semana ISO para ajuste:", semanas, index=len(semanas)-1)
k = st.slider("Número de semanas históricas para suavizar", 1, min(4, len(semanas)-1), 3)
alpha = st.slider("Peso para la semana seleccionada", 0.0, 1.0, 0.6)

# desviación de la semana seleccionada
df_cur = df[df['semana_iso']==sem_sel]
cur = df_cur.groupby(['dia_semana','intervalo'])['desvio_%'].mean().reset_index().rename(columns={'desvio_%':'desvio_cur'})

# promedio de las k semanas anteriores
prev_weeks = [w for w in semanas if w < sem_sel][-k:]
df_prev = df[df['semana_iso'].isin(prev_weeks)]
prev = df_prev.groupby(['dia_semana','intervalo'])['desvio_%'].mean().reset_index().rename(columns={'desvio_%':'desvio_prev'})

# combinar y calcular ajuste
aj = pd.merge(cur, prev, on=['dia_semana','intervalo'], how='left').fillna(0)
aj['desvio_comb'] = alpha * aj['desvio_cur'] + (1-alpha) * aj['desvio_prev']
aj['ajuste_sugerido'] = (1 - aj['desvio_comb']/100).round(4)

st.write(f"**Semana ISO {sem_sel}**, usando promedio de semanas {prev_weeks}")
st.dataframe(aj[['dia_semana','intervalo','desvio_cur','desvio_prev','desvio_comb','ajuste_sugerido']], use_container_width=True)
csv_aj = aj.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Descargar ajustes (.csv)",
    data=csv_aj,
    file_name=f"ajustes_semana_{sem_sel}.csv",
    mime="text/csv"
)

# ─────────── 4. KPIs de Error ───────────
st.subheader("🔢 KPIs de Planificación vs. Realidad")
y_true_all, y_pred_all = serie_continua['reales'], serie_continua['planificados']
mae_all  = mean_absolute_error(y_true_all, y_pred_all)
rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
mape_all = np.mean(np.abs((y_true_all - y_pred_all) / y_true_all.replace(0, np.nan))) * 100

y_true_w, y_pred_w = serie_last['reales'], serie_last['planificados']
mae_w  = mean_absolute_error(y_true_w, y_pred_w)
rmse_w = np.sqrt(mean_squared_error(y_true_w, y_pred_w))
mape_w = np.mean(np.abs((y_true_w - y_pred_w) / y_true_w.replace(0, np.nan))) * 100

st.markdown(
    f"- **MAE:** Total={mae_all:.0f}, Semana={mae_w:.0f}  \n"
    f"- **RMSE:** Total={rmse_all:.0f}, Semana={rmse_w:.0f}  \n"
    f"- **MAPE:** Total={mape_all:.2f}%, Semana={mape_w:.2f}%"
)

st.subheader("💡 Recomendaciones")
if mape_all > 20:
    st.warning("MAPE global >20%: revisar intervalos con mayor desviación.")
elif mape_w > mape_all:
    st.info(f"MAPE semana ({mape_w:.2f}%) > global ({mape_all:.2f}%). Investigar cambios.")
else:
    st.success("Buen alineamiento planificado vs real.")

# ─────────── 5. Tabla de errores por intervalo ───────────
st.subheader("📋 Intervalos con mayor error")
op = st.selectbox("Mostrar errores de:", ["Total","Última Semana"])
errors = serie_continua.copy() if op=="Total" else serie_last.copy()
errors['error_abs'] = (errors['reales'] - errors['planificados']).abs()
errors['MAPE'] = (errors['error_abs'] / errors['planificados'].replace(0, np.nan)) * 100

tab = (
    errors.reset_index()[['_dt','planificados','reales','error_abs','MAPE']]
    .assign(
        MAPE=lambda d: d['MAPE'].map(lambda x: f"{x:.2f}%"),
        error_abs=lambda d: d['error_abs'].astype(int)
    )
)
st.markdown("**MAPE** = |reales - planificados| / planificados × 100")
st.dataframe(tab.sort_values('MAPE', ascending=False).head(10), use_container_width=True)

# ─────────── 6. Heatmap de desvíos ───────────
st.subheader("🔥 Heatmap: Desvío % por Intervalo y Día de la Semana")
heat = df.pivot_table(index='intervalo', columns='dia_semana', values='desvio_%', aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title('Heatmap Desvío %')
st.pyplot(fig3)

# ─────────── 7. Vistas interactivas con anomalías ───────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ['Día','Semana','Mes'])
if vista=='Día':
    fig = px.line(
        serie_continua.reset_index(), x='_dt', y=['planificados','reales'],
        title='📅 Contactos Día',
        labels={'_dt':'Fecha y Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'}
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    decomp = seasonal_decompose(serie_continua['planificados'], model='additive', period=48)
    resid = decomp.resid.dropna()
    anoms = resid[np.abs(resid) > 3*resid.std()]
    fig_anom = px.line(
        serie_continua.reset_index(), x='_dt', y='planificados',
        title='🔴 Anomalías Día',
        color_discrete_map={'planificados':'orange'}
    )
    fig_anom.add_scatter(
        x=anoms.index, y=serie_continua.loc[anoms.index,'planificados'],
        mode='markers', marker=dict(color='red'), name='Anom'
    )
    st.plotly_chart(fig_anom, use_container_width=True)

elif vista=='Semana':
    fig = px.line(
        serie_last.reset_index(), x='_dt', y=['planificados','reales'],
        title=f'📆 Contactos Semana ISO {ultima_sem}',
        labels={'_dt':'Fecha y Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'}
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    decomp = seasonal_decompose(serie_last['planificados'], model='additive', period=48)
    resid = decomp.resid.dropna()
    anoms = resid[np.abs(resid) > 3*resid.std()]
    fig_anom = px.line(
        serie_last.reset_index(), x='_dt', y='planificados',
        title='🔴 Anomalías Semana',
        color_discrete_map={'planificados':'orange'}
    )
    fig_anom.add_scatter(
        x=anoms.index, y=serie_last.loc[anoms.index,'planificados'],
        mode='markers', marker=dict(color='red'), name='Anom'
    )
    st.plotly_chart(fig_anom, use_container_width=True)

else:  # Mes
    daily_m = (
        df.assign(dia=df['fecha'].dt.date)
          .groupby(['nombre_mes','dia','intervalo'])[['planificados','reales']]
          .sum()
          .reset_index()
    )
    monthly_avg = (
        daily_m
        .groupby(['nombre_mes','intervalo'])[['planificados','reales']]
        .mean()
        .reset_index()
    )
    fig = px.line(
        monthly_avg, x='intervalo', y=['planificados','reales'],
        facet_col='nombre_mes', facet_col_wrap=3,
        title='📊 Curva Horaria Promedio Diario por Mes',
        labels={'intervalo':'Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'}
    )
    fig.update_layout(hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
