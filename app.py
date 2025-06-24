import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# ─────────── Configuración de página ───────────
st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes + KPIs, Anomalías y Descomposición")

# ─────────── 1. Carga de datos ───────────
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv","xlsx"])
if not file:
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

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

dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# ─────────── 2.1 Serie continua (fecha + hora) ───────────
df['dt'] = df.apply(lambda r: datetime.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_continua = df.groupby('dt')[['planificados','reales']].sum().sort_index()

# ─────────── 2.2 Última semana de datos ───────────
last_week = df['semana_iso'].max()
df_last = df[df['semana_iso'] == last_week]
serie_last = df_last.groupby('dt')[['planificados','reales']].sum().sort_index()

# ─────────── 2.3 Proyección de ajustes última semana ───────────
aj = (
    df_last.groupby(['dia_semana','intervalo'])['desvio_%']
          .mean()
          .reset_index()
)
aj['ajuste_sugerido'] = aj['desvio_%'].round(2) / 100
aj['semana_obj'] = f"Semana ISO {last_week}"
st.subheader(f"📆 Ajustes sugeridos para la Semana ISO {last_week}")
st.dataframe(aj, use_container_width=True)

# ─────────── 3. KPIs de Error ───────────
st.subheader("🔢 KPIs de Error")
# KPI global
y_true_all = serie_continua['reales']
y_pred_all = serie_continua['planificados']
mae_all  = mean_absolute_error(y_true_all, y_pred_all)
rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
mape_all = np.mean(np.abs((y_true_all - y_pred_all) / y_pred_all.replace(0, np.nan))) * 100

# KPI última semana
y_true_w = serie_last['reales']
y_pred_w = serie_last['planificados']
mae_w  = mean_absolute_error(y_true_w, y_pred_w)
rmse_w = np.sqrt(mean_squared_error(y_true_w, y_pred_w))
mape_w = np.mean(np.abs((y_true_w - y_pred_w) / y_pred_w.replace(0, np.nan))) * 100

st.markdown(
    f"- **MAE (Total):** {mae_all:,.0f} contactos  |  **MAE (Semana {last_week}):** {mae_w:,.0f}  \\"
    f"- **RMSE (Total):** {rmse_all:,.0f} contactos  |  **RMSE (Semana {last_week}):** {rmse_w:,.0f}  \\"
    f"- **MAPE (Total):** {mape_all:.2f}%  |  **MAPE (Semana {last_week}):** {mape_w:.2f}%"
)

# Interpretación inteligente de KPIs
if mape_all > 20:
    st.warning("El MAPE global es superior al 20%, lo que indica un grado importante de error porcentual. Revisa los intervalos con mayor error.")
elif mape_w > mape_all:
    st.info(f"El error porcentual en la última semana ({mape_w:.2f}%) superó al global ({mape_all:.2f}%). Puede haber cambios recientes en el comportamiento.")
else:
    st.success("El desempeño de planificación se mantiene estable en la última semana comparado con el histórico.")

# ─────────── 4. Detección de Anomalías (Data Set Completo) ───────────
st.subheader("🔍 Anomalías (|Residuo| > 3σ) - Data Set Completo")
period = df['intervalo'].nunique()
decomp_all = seasonal_decompose(serie_continua['planificados'], model='additive', period=period, extrapolate_trend='freq')
resid_all = decomp_all.resid.dropna()
sigma_all = resid_all.std()
anoms_all = resid_all[np.abs(resid_all) > 3 * sigma_all]

fig_anom = px.line(
    serie_continua.reset_index(), x='dt', y='planificados',
    labels={'dt':'Fecha-Hora','planificados':'Planificados'},
    title="🔴 Anomalías en Planificados (Historico Completo)"
)
fig_anom.add_scatter(
    x=anoms_all.index, y=serie_continua.loc[anoms_all.index,'planificados'],
    mode='markers', marker=dict(color='red', size=6), name='Anomalías'
)
fig_anom.update_layout(hovermode="x unified")
st.plotly_chart(fig_anom, use_container_width=True, config={"scrollZoom":True})

# ─────────── 5. Descomposición de Serie Temporal ───────────
st.subheader("🔎 Descomposición Serie Temporal - Planificados")
fig_dec, axes = plt.subplots(4, 1, figsize=(12,10), sharex=True)
axes[0].plot(decomp_all.observed);  axes[0].set_ylabel("Observado")
axes[1].plot(decomp_all.trend);     axes[1].set_ylabel("Tendencia")
axes[2].plot(decomp_all.seasonal);  axes[2].set_ylabel("Estacional")
axes[3].plot(decomp_all.resid);      axes[3].set_ylabel("Residuo")
axes[3].set_xlabel("Fecha y Hora")
fig_dec.tight_layout()
st.pyplot(fig_dec)

# ─────────── 6. Selector de Vista ───────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ["Día","Semana","Mes"])

# ─────────── 7. Vistas Interactivas ───────────
if vista == "Día":
    fig_view = px.line(
        serie_continua.reset_index(), x='dt', y=['planificados','reales'],
        labels={'dt':'Fecha y Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📅 Contactos por Intervalo (Fecha + Hora)"
    )
elif vista == "Semana":
    # Totales y curva promedio diario última semana
    sem_totales = df_last.groupby('semana_iso')[['planificados','reales']].sum().reset_index()
    st.bar_chart(sem_totales.set_index('semana_iso'))
    daily = df_last.assign(dia=df_last['fecha'].dt.date).groupby(['dia','intervalo'])[['planificados','reales']].sum().reset_index()
    weekly_avg = daily.groupby('intervalo')[['planificados','reales']].mean().reset_index()
    fig_view = px.line(weekly_avg, x='intervalo', y=['planificados','reales'],
                       labels={'intervalo':'Hora','value':'Promedio diario','variable':'Tipo'},
                       title=f"📆 Curva horaria promedio diario Semana ISO {last_week}")
else:
    # Totales mensuales y curva promedio diario mensual
    monthly_tot = df.groupby('nombre_mes')[['planificados','reales']].sum().rename_axis('Mes').reset_index()
    st.bar_chart(monthly_tot.set_index('Mes'))
    daily_month = df.assign(dia=df['fecha'].dt.date).groupby(['nombre_mes','dia','intervalo'])[['planificados','reales']].sum().reset_index()
    monthly_avg = daily_month.groupby(['nombre_mes','intervalo'])[['planificados','reales']].mean().reset_index()
    fig_view = px.line(monthly_avg, x='intervalo', y=['planificados','reales'], facet_col='nombre_mes', facet_col_wrap=3,
                       labels={'intervalo':'Hora','value':'Promedio diario','variable':'Tipo','nombre_mes':'Mes'},
                       title="🌙 Curva horaria promedio diario por Mes")

fig_view.update_traces(line=dict(width=2))
fig_view.update_xaxes(tickformat='%H:%M', fixedrange=False)
fig_view.update_layout(hovermode="x unified")
st.plotly_chart(fig_view, use_container_width=True, config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

# ─────────── 8. Análisis adicional ───────────
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
heat = df.pivot_table(values='desvio_%', index='intervalo', columns='dia_semana', aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap % Desvío")
st.pyplot(fig3)
