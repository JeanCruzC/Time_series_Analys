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
aj['ajuste_sugerido'] = (aj['desvio_%'].round(2)) / 100
aj['semana_obj'] = f"Semana ISO {last_week}"
st.subheader(f"📆 Ajustes sugeridos para la Semana ISO {last_week}")
st.dataframe(aj, use_container_width=True)

# ─────────── 3. Selector de Vista ───────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ["Día","Semana","Mes"])

# ────────────────────────────────────────────────
# 3.1 VISTA DÍA (sumas reales vs planificados) ────────────────────────────────────────────────
if vista == "Día":
    fig_day = px.line(
        serie_continua.reset_index(), x='dt', y=['planificados','reales'],
        labels={'dt':'Fecha y Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
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
        ]),
        type="date", fixedrange=False
    )
    fig_day.update_layout(hovermode="x unified", dragmode="zoom", yaxis=dict(autorange=True))
    st.plotly_chart(fig_day, use_container_width=True, config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

# ────────────────────────────────────────────────
# 3.2 VISTA SEMANA (sumas semanales vs curva horaria promedio diario) ────────────────────────────────────────────────
elif vista == "Semana":
    # Totales semanales
    sem_totales = df_last.groupby('semana_iso')[['planificados','reales']].sum().reset_index()
    st.markdown(f"**Totales Semana ISO {last_week}**")
    st.bar_chart(sem_totales.set_index('semana_iso'))
    
    # Curva horaria promedio diario
    daily = (
        df_last.assign(dia=df_last['fecha'].dt.date)
               .groupby(['dia','intervalo'])[['planificados','reales']]
               .sum()
               .reset_index()
    )
    weekly_avg = daily.groupby('intervalo')[['planificados','reales']].mean().reset_index()
    fig_week = px.line(
        weekly_avg, x='intervalo', y=['planificados','reales'],
        labels={'intervalo':'Hora','value':'Promedio diario','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📆 Curva horaria promedio diario Semana ISO {last_week}"
    )
    fig_week.update_traces(line=dict(width=2))
    fig_week.update_xaxes(tickformat='%H:%M', fixedrange=False)
    fig_week.update_layout(hovermode="x unified")
    st.plotly_chart(fig_week, use_container_width=True, config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

# ────────────────────────────────────────────────
# 3.3 VISTA MES: totales mensuales vs curva horaria promedio diario
# ────────────────────────────────────────────────
else:
    # Totales por mes\m    monthly = df.groupby(['mes','nombre_mes'])[['planificados','reales']].sum().reset_index()
    st.markdown("**Totales por Mes**")
    st.bar_chart(monthly.set_index('nombre_mes')[['planificados','reales']])

    # Curva horaria promedio diario por mes
    daily_month = (
        df.assign(dia=df['fecha'].dt.date)
          .groupby(['nombre_mes','dia','intervalo'])[['planificados','reales']]
          .sum()
          .reset_index()
    )
    monthly_avg = daily_month.groupby(['nombre_mes','intervalo'])[['planificados','reales']].mean().reset_index()
    fig_mon = px.line(
        monthly_avg, x='intervalo', y=['planificados','reales'],
        facet_col='nombre_mes', facet_col_wrap=3,
        labels={'intervalo':'Hora','value':'Promedio diario','variable':'Tipo','nombre_mes':'Mes'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="🌙 Curva horaria promedio diario por Mes (00:00–23:59)"
    )
    fig_mon.update_traces(line=dict(width=2))
    fig_mon.update_xaxes(tickformat='%H:%M', fixedrange=False)
    fig_mon.update_layout(showlegend=False, hovermode="x unified")
    fig_mon.for_each_trace(lambda t: t.update(showlegend=True) if t.name=='reales' else None)
    st.plotly_chart(fig_mon, use_container_width=True, config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

# ────────────────────────────────────────────────
# 4. Desvío Promedio por Intervalo
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

# ────────────────────────────────────────────────
# 5. Heatmap de Desvío
# ────────────────────────────────────────────────
st.subheader("🔥 Heatmap: Desvío por Día de la Semana y Intervalo")
heat = df.pivot_table(values='desvio_%', index='intervalo', columns='dia_semana', aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap % Desvío")
st.pyplot(fig3)

# ────────────────────────────────────────────────
# 6. KPIs de Error por última semana (MAE, RMSE, MAPE)
# ────────────────────────────────────────────────
kp = serie_last.reset_index()
y_true = kp['reales']
y_pred = kp['planificados']

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_pred.replace(0, np.nan))) * 100

st.subheader("🔢 KPIs de Error - Última Semana (ISO {last_week})")
st.markdown(f"- **MAE:** {mae:,.0f} contactos  \n"
            f"- **RMSE:** {rmse:,.0f} contactos  \n"
            f"- **MAPE:** {mape:.2f}%")

# ────────────────────────────────────────────────
# 7. Detección de Anomalías última semana
# ────────────────────────────────────────────────
period = df['intervalo'].nunique()
decomp_last = seasonal_decompose(serie_last['planificados'], model='additive', period=period, extrapolate_trend='freq')
resid_last = decomp_last.resid.dropna()
sigma_last = resid_last.std()
anoms_last = resid_last[np.abs(resid_last) > 3*sigma_last]

fig_anom = px.line(serie_last.reset_index(), x='dt', y='planificados', labels={'dt':'Fecha-Hora','planificados':'Planificados'}, title="🔴 Anomalías - Última Semana")
fig_anom.add_scatter(x=anoms_last.index, y=serie_last.loc[anoms_last.index,'planificados'], mode='markers', marker=dict(color='red', size=6), name='Anomalías')
fig_anom.update_layout(hovermode="x unified")
st.plotly_chart(fig_anom, use_container_width=True, config={"scrollZoom":True})

# ────────────────────────────────────────────────
# 8. Descomposición última semana
# ────────────────────────────────────────────────
st.subheader("🔍 Descomposición - Última Semana (Planificados)")
fig_dec, axs = plt.subplots(4,1, figsize=(12,10), sharex=True)
axs[0].plot(decomp_last.observed);  axs[0].set_ylabel("Observado")
axs[1].plot(decomp_last.trend);     axs[1].set_ylabel("Tendencia")
axs[2].plot(decomp_last.seasonal);  axs[2].set_ylabel("Estacional")
axs[3].plot(deomp_last.resid);      axs[3].set_ylabel("Residuo")
axs[3].set_xlabel("Fecha y Hora")
fig_dec.tight_layout()
st.pyplot(fig_dec)
