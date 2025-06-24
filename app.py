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

df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes'] = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')

df['desvio'] = df['reales'] - df['planificados']
df['desvio_%'] = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100

# Orden de días
orden_dias = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=orden_dias, ordered=True)

# Serie continua
import datetime as dt
	df['dt'] = df.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_continua = df.groupby('dt')[['planificados','reales']].sum().sort_index()

# Última semana
ultima_sem = df['semana_iso'].max()
df_last = df[df['semana_iso']==ultima_sem]
serie_last = df_last.groupby('dt')[['planificados','reales']].sum().sort_index()

# Ajustes sugeridos última semana
ajustes = (
    df_last.groupby(['dia_semana','intervalo'])['desvio_%']
          .mean().reset_index()
)
ajustes['ajuste_sugerido'] = ajustes['desvio_%'].round(2)/100
ajustes['semana'] = f"Semana ISO {ultima_sem}"
st.subheader(f"📆 Ajustes sugeridos - Semana ISO {ultima_sem}")
st.dataframe(ajustes, use_container_width=True)

# KPIs global y última semana
st.subheader("🔢 KPIs de Planificación vs. Realidad")
# global
y_true_all = serie_continua['reales']
y_pred_all = serie_continua['planificados']
mae_all = mean_absolute_error(y_true_all, y_pred_all)
rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
mape_all = np.mean(np.abs((y_true_all - y_pred_all)/y_pred_all.replace(0,np.nan)))*100
# última semana
y_true_w = serie_last['reales']
y_pred_w = serie_last['planificados']
mae_w = mean_absolute_error(y_true_w, y_pred_w)
rmse_w = np.sqrt(mean_squared_error(y_true_w, y_pred_w))
mape_w = np.mean(np.abs((y_true_w - y_pred_w)/y_pred_w.replace(0,np.nan)))*100

st.markdown(
    f"- **MAE:** Total={mae_all:.0f}, Semana={mae_w:.0f}  \\"
    f"- **RMSE:** Total={rmse_all:.0f}, Semana={rmse_w:.0f}  \\"
    f"- **MAPE:** Total={mape_all:.2f}%, Semana={mape_w:.2f}%"
)

# Recomendaciones automáticas
st.subheader("💡 Recomendaciones")
if mape_all>20:
    st.warning("El error porcentual global (MAPE) supera 20%. Se recomienda revisar los intervalos con desviaciones mayores y ajustar planificados en esos rangos.")
elif mape_w>mape_all:
    st.info(f"El desempeño en la última semana ({mape_w:.2f}%) empeoró respecto al histórico ({mape_all:.2f}%). Investigar cambios recientes en la operación.")
else:
    st.success("La planificación está alineada con el comportamiento real. Mantener parámetros actuales.")

# Tabla de intervalos con mayor error y slicer
st.subheader("📋 Errores por Intervalo")
opcion = st.selectbox("Mostrar errores de:", ["Total","Última Semana"])
if opcion=="Total":
    kd = serie_continua.copy()
else:
    kd = serie_last.copy()
kd['error_abs'] = np.abs(kd['reales'] - kd['planificados'])
kd['error_pct'] = np.abs((kd['reales'] - kd['planificados'])/kd['planificados'].replace(0,np.nan))*100
kd_tab = kd.reset_index().groupby('dt')[['error_abs','error_pct']].mean()
st.dataframe(kd_tab.sort_values('error_abs', ascending=False).head(10))

# Heatmap de desvíos original
st.subheader("🔥 Heatmap: Desvío % por Intervalo y Día de la Semana")
heat = df.pivot(index='intervalo', columns='dia_semana', values='desvio_%')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax3)
ax3.set_title("Heatmap Desvío %")
st.pyplot(fig3)

# Vistas interactivas manteniendo código anterior...
