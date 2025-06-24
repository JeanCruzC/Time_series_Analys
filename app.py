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
# Tipos
df['fecha'] = pd.to_datetime(df['fecha'])
df['intervalo'] = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
# Campos
# ... (igual que antes)
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes'] = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio'] = df['reales'] - df['planificados']
df['desvio_%'] = (df['desvio'] / df['planificados'].replace(0, np.nan)) * 100
dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# Serie continua
_df = df.copy()
_df['dt'] = _df.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_continua = _df.groupby('dt')[['planificados','reales']].sum().sort_index()

# Última semana# Última semana\ultima_sem = df['semana_iso'].max()
df_last = df[df['semana_iso']==ultima_sem].copy()
_df_last = df_last.copy()
_df_last['dt'] = _df_last.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_last = _df_last.groupby('dt')[['planificados','reales']].sum().sort_index()

# Ajustes sugeridos
ajustes = df_last.groupby(['dia_semana','intervalo'])['desvio_%'].mean().reset_index()
ajustes['ajuste_sugerido'] = ajustes['desvio_%'].round(2)/100
st.subheader(f"📆 Ajustes sugeridos - Semana ISO {ultima_sem}")
st.dataframe(ajustes, use_container_width=True)

# KPIs global y última semana
st.subheader("🔢 KPIs de Planificación vs. Realidad")
y_true_all, y_pred_all = serie_continua['reales'], serie_continua['planificados']
y_true_w, y_pred_w = serie_last['reales'], serie_last['planificados']
mae_all = mean_absolute_error(y_true_all,y_pred_all); rmse_all = np.sqrt(mean_squared_error(y_true_all,y_pred_all)); mape_all = np.mean(np.abs((y_true_all-y_pred_all)/y_pred_all.replace(0,np.nan)))*100
mae_w = mean_absolute_error(y_true_w,y_pred_w); rmse_w = np.sqrt(mean_squared_error(y_true_w,y_pred_w)); mape_w = np.mean(np.abs((y_true_w-y_pred_w)/y_pred_w.replace(0,np.nan)))*100
st.markdown(f"- MAE: Total={mae_all:.0f}, Semana={mae_w:.0f}  \n- RMSE: Total={rmse_all:.0f}, Semana={rmse_w:.0f}  \n- MAPE: Total={mape_all:.2f}%, Semana={mape_w:.2f}%")
# Recomendaciones
st.subheader("💡 Recomendaciones")
if mape_all>20: st.warning("MAPE global >20%: revisar intervalos de mayor error.")
elif mape_w>mape_all: st.info(f"MAPE semana ({mape_w:.2f}%) > global ({mape_all:.2f}%). Revisar cambios.")
else: st.success("Buen alineamiento planificado vs real.")

# Heatmap de los desvíos (pivot_table)
st.subheader("🔥 Heatmap: Desvío % por Intervalo y Día de la Semana")
heat = df.pivot_table(index='intervalo', columns='dia_semana', values='desvio_%', aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(heat, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title('Heatmap Desvío %')
st.pyplot(fig3)

# Selector y Vistas interactivas con anomalías reinsertadas
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:",['Día','Semana','Mes'])
if vista=='Día':
    fig = px.line(serie_continua.reset_index(),x='dt',y=['planificados','reales'],title='📅 Contactos Día')
    st.plotly_chart(fig,use_container_width=True)
    # anomalías día
    decomp = seasonal_decompose(serie_continua['planificados'],model='additive',period=serie_continua.index.get_level_values(0).nunique())
    resid = decomp.resid.dropna(); sigma=resid.std(); anoms=resid[np.abs(resid)>3*sigma]
    fig_anom=px.line(serie_continua.reset_index(),x='dt',y='planificados',title='🔴 Anomalías Día')
    fig_anom.add_scatter(x=anoms.index,y=serie_continua.loc[anoms.index,'planificados'],mode='markers',marker=dict(color='red'),name='Anom')
    st.plotly_chart(fig_anom,use_container_width=True)
elif vista=='Semana':
    fig = px.line(serie_last.reset_index(),x='dt',y=['planificados','reales'],title='📆 Contactos Última Semana')
    st.plotly_chart(fig,use_container_width=True)
    # anomalías semana
    decomp = seasonal_decompose(serie_last['planificados'],model='additive',period=serie_last.index.get_level_values(0).nunique())
    resid=decomp.resid.dropna(); sigma=resid.std(); anoms=resid[np.abs(resid)>3*sigma]
    fig_anom=px.line(serie_last.reset_index(),x='dt',y='planificados',title='🔴 Anomalías Semana')
    fig_anom.add_scatter(x=anoms.index,y=serie_last.loc[anoms.index,'planificados'],mode='markers',marker=dict(color='red'),name='Anom')
    st.plotly_chart(fig_anom,use_container_width=True)
else:
    # agregamos proyección mensual promedio diario
    daily_m = df.assign(dia=df['fecha'].dt.date).groupby(['nombre_mes','dia','intervalo'])[['planificados','reales']].sum()
    monthly_avg=daily_m.groupby(['nombre_mes','intervalo']).mean().reset_index()
    fig=px.line(monthly_avg,x='intervalo',y=['planificados','reales'],facet_col='nombre_mes',facet_col_wrap=3,title='📊 Curva Horaria Mes')
    st.plotly_chart(fig,use_container_width=True)
    # anomalías mes (usamos planificados de monthly_avg)
    ts=pd.Series(monthly_avg['planificados'].values,index=pd.to_datetime(monthly_avg['intervalo'].astype(str),format='%H:%M'))
    decomp=seasonal_decompose(ts,model='additive',period=24)
    resid=decomp.resid.dropna(); sigma=resid.std(); anoms=resid[np.abs(resid)>3*sigma]
    fig_anom=px.line(ts.reset_index(),x='index',y=0,title='🔴 Anomalías Mes')
    fig_anom.add_scatter(x=anoms.index,y=ts.loc[anoms.index],mode='markers',marker=dict(color='red'),name='Anom')
    st.plotly_chart(fig_anom,use_container_width=True)
