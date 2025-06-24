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

# ─────────── 3. Selector de Vista ───────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ["Día","Semana","Mes"])

# ────────────────────────────────────────────────
# 3.1 VISTA DÍA (sumas) ────────────────────────────────────────────────
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
# 3.2 VISTA SEMANA: anomalías y KPIs no aplican en vista, solo curva horaria sumas ────────────────────────────────────────────────
elif vista == "Semana":
    weekly_detail = df.groupby(['semana_iso','intervalo'])[['planificados','reales']].sum().reset_index()
    fig_week = px.line(
        weekly_detail, x='intervalo', y=['planificados','reales'],
        facet_col='semana_iso', facet_col_wrap=4,
        labels={'intervalo':'Hora','value':'Volumen','variable':'Tipo','semana_iso':'Semana ISO'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📆 Curva horaria por Semana (00:00–23:59)"
    )
    fig_week.update_traces(line=dict(width=2))
    fig_week.update_xaxes(tickformat='%H:%M', matches=None, fixedrange=False)
    fig_week.update_layout(showlegend=False, hovermode="x unified")
    fig_week.for_each_trace(lambda t: t.update(showlegend=True) if t.name=='reales' else None)
    st.plotly_chart(fig_week,use_container_width=True,config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

# ────────────────────────────────────────────────
# 3.3 VISTA MES: sumas + curva horaria promedio diario ────────────────────────────────────────────────
else:
    # Totales mensuales
    monthly = df.groupby(['mes','nombre_mes'])[['planificados','reales']].sum().reset_index()
    fig_mon = px.line(
        monthly, x='nombre_mes', y=['planificados','reales'],
        labels={'nombre_mes':'Mes','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📊 Contactos por Mes (Totales)"
    )
    fig_mon.update_traces(line=dict(width=2))
    fig_mon.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_mon,use_container_width=True,config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

    # Curva promedio diario
    days_mes = df.groupby('mes')['fecha'].dt.date.nunique().to_dict()
    daily_month = df.groupby(['nombre_mes','intervalo'])[['planificados','reales']].sum().reset_index()
    daily_month['planificados'] = daily_month.apply(lambda r: r['planificados']/days_mes[df[df['nombre_mes']==r['nombre_mes']]['mes'].iloc[0]], axis=1)
    daily_month['reales'] = daily_month.apply(lambda r: r['reales']/days_mes[df[df['nombre_mes']==r['nombre_mes']]['mes'].iloc[0]], axis=1)
    fig_mon_avg = px.line(
        daily_month, x='intervalo', y=['planificados','reales'],
        facet_col='nombre_mes', facet_col_wrap=2,
        labels={'intervalo':'Hora','value':'Promedio diario','variable':'Tipo','nombre_mes':'Mes'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="🌙 Curva horaria promedio diario por Mes (00:00–23:59)"
    )
    fig_mon_avg.update_traces(line=dict(width=2))
    fig_mon_avg.update_xaxes(tickformat='%H:%M', matches=None, fixedrange=False)
    fig_mon_avg.update_layout(showlegend=False, hovermode="x unified")
    fig_mon_avg.for_each_trace(lambda t: t.update(showlegend=True) if t.name=='reales' else None)
    st.plotly_chart(fig_mon_avg,use_container_width=True,config={"scrollZoom":True,"modeBarButtonsToAdd":["autoScale2d"]})

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
# 6. KPIs de Error (MAE, RMSE, MAPE)
# ────────────────────────────────────────────────
kpi_data = serie_continua.reset_index()
y_true = kpi_data['reales']
y_pred = kpi_data['planificados']

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred)/y_pred.replace(0, np.nan))) * 100

st.subheader("🔢 KPIs de Error por Intervalo")
st.markdown(f"- **MAE:** {mae:,.0f} contactos  \n"
            f"- **RMSE:** {rmse:,.0f} contactos  \n"
            f"- **MAPE:** {mape:.2f}%")

# Gráficas de error
grid = plt.subplots(1,2,figsize=(14,4))
fig_kpi, axes = grid
kpi_data['error_abs'] = np.abs(y_true - y_pred)
kpi_data['error_pct'] = np.abs((y_true - y_pred)/y_pred.replace(0, np.nan)) * 100

axes[0].bar(kpi_data['dt'], kpi_data['error_abs'])
axes[0].set_title("Error Absoluto por Intervalo")
axes[0].set_ylabel("Contactos")
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(kpi_data['dt'], kpi_data['error_pct'])
axes[1].set_title("MAPE por Intervalo")
axes[1].set_ylabel("% Error")
axes[1].tick_params(axis='x', rotation=45)

st.pyplot(fig_kpi)

# ────────────────────────────────────────────────
# 7. Anomalías (residual > 3σ)
# ────────────────────────────────────────────────
period = df['intervalo'].nunique()
decomp = seasonal_decompose(serie_continua['planificados'], model='additive', period=period, extrapolate_trend='freq')
resid = decomp.resid.dropna()
sigma = resid.std()
anoms = resid[np.abs(resid) > 3*sigma]

fig_anom = px.line(serie_continua.reset_index(), x='dt', y='planificados', labels={'dt':'Fecha-Hora','planificados':'Planificados'}, title="🔴 Anomalías en Planificados")
fig_anom.add_scatter(x=anoms.index, y=serie_continua.loc[anoms.index,'planificados'], mode='markers', marker=dict(color='red',size=6), name='Anomalías')
fig_anom.update_layout(hovermode="x unified")
st.plotly_chart(fig_anom,use_container_width=True,config={"scrollZoom":True})

# ────────────────────────────────────────────────
# 8. Descomposición de Serie Temporal
# ────────────────────────────────────────────────
st.subheader("🔍 Descomposición de Serie Temporal (Planificados)")
fig_dec, axes = plt.subplots(4,1,figsize=(12,10),sharex=True)
axes[0].plot(decomp.observed);  axes[0].set_ylabel("Observado")
axes[1].plot(decomp.trend);     axes[1].set_ylabel("Tendencia")
axes[2].plot(decomp.seasonal);  axes[2].set_ylabel("Estacional")
axes[3].plot(decomp.resid);     axes[3].set_ylabel("Residuo")
axes[3].set_xlabel("Fecha y Hora")
fig_dec.tight_layout()
st.pyplot(fig_dec)
