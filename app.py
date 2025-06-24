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
    'fecha': 'fecha', 'tramo': 'intervalo',
    'planif. contactos': 'planificados', 'contactos': 'reales'
})
# Conversión de tipos
df['fecha'] = pd.to_datetime(df['fecha'])
df['intervalo'] = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
# Campos adicionales
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes'] = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
# Desvíos
df['desvio'] = df['reales'] - df['planificados']
df['desvio_%'] = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100
# Orden de días
dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# ─────────── 2.1 Serie continua ───────────
df['dt'] = df.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_continua = df.groupby('dt')[['planificados','reales']].sum().sort_index()

# ─────────── 2.2 Última semana de datos ───────────
ultima_sem = df['semana_iso'].max()
df_last = df[df['semana_iso'] == ultima_sem]
serie_last = df_last.groupby('dt')[['planificados','reales']].sum().sort_index()

# ─────────── 2.3 Ajustes sugeridos ───────────
ajustes = df_last.groupby(['dia_semana','intervalo'])['desvio_%'].mean().reset_index()
ajustes['ajuste_sugerido'] = ajustes['desvio_%'].round(2) / 100
ajustes['semana'] = f"Semana ISO {ultima_sem}"
st.subheader(f"📆 Ajustes sugeridos - Semana ISO {ultima_sem}")
st.dataframe(ajustes, use_container_width=True)

# ─────────── 3. KPIs de Error ───────────
st.subheader("🔢 KPIs de Planificación vs. Realidad")
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
    f"- **MAE:** Total={mae_all:.0f}, Semana={mae_w:.0f}  \n"
    f"- **RMSE:** Total={rmse_all:.0f}, Semana={rmse_w:.0f}  \n"
    f"- **MAPE:** Total={mape_all:.2f}%, Semana={mape_w:.2f}%"
)

# Recomendaciones automáticas
st.subheader("💡 Recomendaciones")
if mape_all > 20:
    st.warning(
        "El MAPE global supera 20%. Se recomienda revisar los intervalos con mayor desviación "
        "y ajustar los planificados en esos franjas horarias."
    )
elif mape_w > mape_all:
    st.info(
        f"El MAPE de la última semana ({mape_w:.2f}%) excede al global ({mape_all:.2f}%). "
        "Investigar cambios recientes en la operación o eventos fuera de lo habitual."
    )
else:
    st.success("La planificación semanal está alineada con el histórico. Mantener parámetros actuales.")

# ─────────── 4. Tabla de errores por intervalo ───────────
st.subheader("📋 Intervalos con mayor error")
opcion = st.selectbox("Mostrar errores de:", ["Total","Última Semana"])
if opcion == "Total":
    errors = serie_continua.copy()
else:
    errors = serie_last.copy()
errors['error_abs'] = np.abs(errors['reales'] - errors['planificados'])
errors['error_pct'] = np.abs((errors['reales'] - errors['planificados']) / errors['planificados'].replace(0, np.nan)) * 100
errors_tab = errors.reset_index().groupby('dt')[['error_abs','error_pct']].mean()
st.dataframe(errors_tab.sort_values('error_abs', ascending=False).head(10))

# ─────────── 5. Heatmap de desvíos ───────────
st.subheader("🔥 Heatmap: Desvío % por Intervalo y Día de la Semana")
heat = df.pivot(index='intervalo', columns='dia_semana', values='desvio_%')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap Desvío %")
st.pyplot(fig3)

# ─────────── 6. Detección de anomalías (completo) ───────────
st.subheader("🔍 Anomalías en Planificados (Completo)")
period = df['intervalo'].nunique()
decomp_all = seasonal_decompose(serie_continua['planificados'], model='additive', period=period, extrapolate_trend='freq')
resid_all = decomp_all.resid.dropna()
sigma_all = resid_all.std()
anoms_all = resid_all[np.abs(resid_all) > 3 * sigma_all]
fig_anom = px.line(
    serie_continua.reset_index(), x='dt', y='planificados',
    labels={'dt':'Fecha-Hora','planificados':'Planificados'},
    title="🔴 Anomalías en Planificados"
)
fig_anom.add_scatter(
    x=anoms_all.index, y=serie_continua.loc[anoms_all.index,'planificados'],
    mode='markers', marker=dict(color='red', size=6), name='Anomalías'
)
st.plotly_chart(fig_anom, use_container_width=True)

# ─────────── 7. Descomposición de la serie temporal ───────────
st.subheader("🔎 Descomposición Serie Temporal - Planificados")
fig_dec, axs = plt.subplots(4,1, figsize=(12,10), sharex=True)
axs[0].plot(decomp_all.observed);  axs[0].set_ylabel("Observado")
axs[1].plot(decomp_all.trend);     axs[1].set_ylabel("Tendencia")
axs[2].plot(decomp_all.seasonal);  axs[2].set_ylabel("Estacional")
axs[3].plot(decomp_all.resid);      axs[3].set_ylabel("Residuo")
axs[3].set_xlabel("Fecha y Hora")
fig_dec.tight_layout()
st.pyplot(fig_dec)

# ─────────── 8. Vistas interactivas ───────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ["Día","Semana","Mes"])
if vista == "Día":
    fig_view = px.line(
        serie_continua.reset_index(), x='dt', y=['planificados','reales'],
        labels={'dt':'Fecha y Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'orange','reales':'blue'},
        title="📅 Contactos por Intervalo (Fecha + Hora)"
    )
elif vista == "Semana":
    st.bar_chart(df_last.groupby('semana_iso')[['planificados','reales']].sum().rename_axis('Semana').reset_index().set_index('Semana'))
    daily = df_last.assign(dia=df_last['fecha'].dt.date).groupby(['dia','intervalo'])[['planificados','reales']].sum().reset_index()
    weekly_avg = daily.groupby('intervalo')[['planificados','reales']].mean().reset_index()
    fig_view = px.line(
        weekly_avg, x='intervalo', y=['planificados','reales'],
        labels={'intervalo':'Hora','value':'Promedio diario','variable':'Tipo'},
        title=f"📆 Curva horaria promedio diario Semana ISO {ultima_sem}"
    )
else:
    st.bar_chart(df.groupby('nombre_mes')[['planificados','reales']].sum().rename_axis('Mes').reset_index().set_index('Mes'))
    daily_m = df.assign(dia=df['fecha'].dt.date).groupby(['nombre_mes','dia','intervalo'])[['planificados','reales']].sum().reset_index()
    monthly_avg = daily_m.groupby(['nombre_mes','intervalo'])[['planificados','reales']].mean().reset_index()
    fig_view = px.line(
        monthly_avg, x='intervalo', y=['planificados','reales'], facet_col='nombre_mes', facet_col_wrap=3,
        labels={'intervalo':'Hora','value':'Promedio diario','variable':'Tipo','nombre_mes':'Mes'}
    )
fig_view.update_traces(line=dict(width=2))
fig_view.update_xaxes(tickformat='%H:%M', fixedrange=False)
fig_view.update_layout(hovermode="x unified")
st.plotly_chart(fig_view, use_container_width=True)
