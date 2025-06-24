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
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv", "xlsx"])
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
df['fecha'] = pd.to_datetime(df['fecha'])
df['intervalo'] = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes'] = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio'] = df['reales'] - df['planificados']
df['desvio_%'] = df['desvio'] / df['planificados'].replace(0, np.nan) * 100

dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# ─────────── 2.1 Serie continua ───────────
df['_dt'] = df.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_continua = df.groupby('_dt')[['planificados', 'reales']].sum().sort_index()

# ─────────── 2.2 Última semana ───────────
ultima_sem = int(df['semana_iso'].max())
_df_last = df[df['semana_iso'] == ultima_sem].copy()
_df_last['_dt'] = _df_last.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_last = _df_last.groupby('_dt')[['planificados', 'reales']].sum().sort_index()

# ─────────── 3. Ajustes sugeridos (combinación ponderada) ───────────
N = 3
alpha = 0.7
proxima_sem = ultima_sem + 1

cur = (_df_last
       .groupby(['dia_semana', 'intervalo'])['desvio_%']
       .mean()
       .reset_index()
       .rename(columns={'desvio_%': 'desvio_cur'}))

prev_weeks = sorted(w for w in df['semana_iso'].unique() if w < ultima_sem)[-N:]
df_prev = df[df['semana_iso'].isin(prev_weeks)]
prev = (df_prev
        .groupby(['dia_semana', 'intervalo'])['desvio_%']
        .mean()
        .reset_index()
        .rename(columns={'desvio_%': 'desvio_prev'}))

aj = pd.merge(cur, prev, on=['dia_semana', 'intervalo'], how='left')
aj['desvio_prev'] = aj['desvio_prev'].fillna(0)
aj['desvio_comb'] = alpha * aj['desvio_cur'] + (1 - alpha) * aj['desvio_prev']
aj['ajuste_sugerido'] = (1 + aj['desvio_comb'] / 100).round(4).map(lambda x: f"{x*100:.0f}%")

st.subheader(f"📆 Ajustes sugeridos para Semana ISO {proxima_sem}")
st.markdown(
    f"**Combinación ponderada:** {int(alpha*100)}% desvío última semana (ISO {ultima_sem}) + "
    f"{int((1-alpha)*100)}% promedio semanas {prev_weeks}"
)
st.markdown("""
**Columnas de la tabla**  
- **dia_semana**: día de la semana (Monday…Sunday)  
- **intervalo**: franja horaria  
- **desvio_cur**: desvío % promedio de la última semana  
- **desvio_prev**: desvío % promedio de las semanas históricas  
- **desvio_comb**: 0.7·desvio_cur + 0.3·desvio_prev  
- **ajuste_sugerido**: factor de ajuste (100% + desvio_comb)
""")
st.dataframe(
    aj[['dia_semana', 'intervalo', 'desvio_cur', 'desvio_prev', 'desvio_comb', 'ajuste_sugerido']],
    use_container_width=True
)
st.download_button("📥 Descargar ajustes (.csv)", data=aj.to_csv(index=False).encode(), file_name=f"ajustes_sem_{proxima_sem}.csv")

# ─────────── 4. KPIs de Error ───────────
st.subheader("🔢 KPIs de Planificación vs. Realidad")
y_t_all, y_p_all = serie_continua['reales'], serie_continua['planificados']
mae_all = mean_absolute_error(y_t_all, y_p_all)
rmse_all = np.sqrt(mean_squared_error(y_t_all, y_p_all))
mape_all = np.mean(np.abs((y_t_all - y_p_all) / y_t_all.replace(0, np.nan))) * 100

y_t_w, y_p_w = serie_last['reales'], serie_last['planificados']
mae_w = mean_absolute_error(y_t_w, y_p_w)
rmse_w = np.sqrt(mean_squared_error(y_t_w, y_p_w))
mape_w = np.mean(np.abs((y_t_w - y_p_w) / y_t_w.replace(0, np.nan))) * 100

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

# ─────────── 4.5 Resumen semanal de Desviación ───────────
st.subheader("🗓️ Resumen semanal de Desviación")
weekly = df.groupby('semana_iso')[['planificados', 'reales']].sum().reset_index()
weekly['desvío_abs'] = weekly['reales'] - weekly['planificados']
weekly['desvío_%'] = weekly['desvío_abs'] / weekly['planificados'].replace(0, np.nan) * 100
weekly['MAPE_%'] = weekly['desvío_abs'].abs() / weekly['planificados'].replace(0, np.nan) * 100
weekly = weekly.rename(columns={'semana_iso': 'Semana ISO'})
weekly['desvío_%'] = weekly['desvío_%'].map(lambda x: f"{x:.2f}%")
weekly['MAPE_%'] = weekly['MAPE_%'].map(lambda x: f"{x:.2f}%")
st.dataframe(
    weekly[['Semana ISO', 'planificados', 'reales', 'desvío_abs', 'desvío_%', 'MAPE_%']],
    use_container_width=True
)

# ─────────── 5. Intervalos con mayor error ───────────
st.subheader("📋 Intervalos con mayor error")
opt = st.selectbox("Mostrar errores de:", ["Total", "Última Semana"])
errors = (serie_continua if opt == "Total" else serie_last).copy()
errors['error_abs'] = (errors['reales'] - errors['planificados']).abs()
errors['MAPE'] = errors['error_abs'] / errors['planificados'].replace(0, np.nan) * 100

tab = (errors.reset_index()[['_dt', 'planificados', 'reales', 'error_abs', 'MAPE']]
       .assign(
           error_abs=lambda d: d['error_abs'].astype(int),
           MAPE=lambda d: d['MAPE'].map(lambda x: f"{x:.2f}%")
       ))
st.markdown("**MAPE** = |reales − planificados| / planificados × 100")
st.dataframe(tab.sort_values('MAPE', ascending=False).head(10), use_container_width=True)

# ─────────── 6. Heatmap ───────────
st.subheader("🔥 Heatmap: Desvío % por Intervalo y Día de la Semana")
heat = df.pivot_table(index='intervalo', columns='dia_semana', values='desvio_%', aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(heat, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title('Heatmap Desvío %')
st.pyplot(fig3)

# ─────────── 7. Vista interactiva con anomalías ───────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ['Día', 'Semana', 'Mes'])

if vista == 'Día':
    fig = px.line(
        serie_continua.reset_index(), x='_dt', y=['planificados', 'reales'],
        title='📅 Contactos Día',
        labels={'_dt': 'Fecha y Hora', 'value':'Volumen', 'variable':'Tipo'},
        color_discrete_map={'planificados':'red','reales':'blue'}
    )
    fig.update_layout(hovermode="x unified", dragmode="zoom")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    decomp = seasonal_decompose(serie_continua['planificados'], model='additive', period=48)
    resid = decomp.resid.dropna()
    anoms = resid[np.abs(resid) > 3*resid.std()]
    fig_anom = px.line(
        serie_continua.reset_index(), x='_dt', y='planificados',
        title='🔴 Anomalías Día',
        labels={'_dt': 'Fecha y Hora', 'planificados':'Planificados'},
        color_discrete_map={'planificados':'red'}
    )
    fig_anom.add_scatter(
        x=anoms.index, y=serie_continua.loc[anoms.index,'planificados'],
        mode='markers', marker=dict(color='red'), name='Anomalías'
    )
    st.plotly_chart(fig_anom, use_container_width=True)

elif vista == 'Semana':
    weekly = df.groupby(['semana_iso','intervalo'])[['planificados','reales']].mean().reset_index()
    melt = weekly.melt(
        id_vars=['semana_iso','intervalo'],
        value_vars=['planificados','reales'],
        var_name='Tipo', value_name='Volumen'
    )
    fig_week = px.line(
        melt, x='intervalo', y='Volumen', color='Tipo',
        animation_frame='semana_iso', animation_group='Tipo',
        labels={'intervalo':'Hora','semana_iso':'Semana ISO','Volumen':'Contactos','Tipo':'Tipo'},
        title="📆 Curvas horarias por Semana",
        color_discrete_map={'planificados':'red','reales':'blue'}
    )
    fig_week.update_layout(hovermode="x unified")
    st.plotly_chart(fig_week, use_container_width=True)

    decomp_w = seasonal_decompose(serie_last['planificados'], model='additive', period=48)
    resid_w = decomp_w.resid.dropna()
    anoms_w = resid_w[np.abs(resid_w) > 3*resid_w.std()]
    fig_anom_w = px.line(
        serie_last.reset_index(), x='_dt', y='planificados',
        title='🔴 Anomalías Semana',
        labels={'_dt': 'Fecha y Hora', 'planificados':'Planificados'},
        color_discrete_map={'planificados':'red'}
    )
    fig_anom_w.add_scatter(
        x=anoms_w.index, y=serie_last.loc[anoms_w.index,'planificados'],
        mode='markers', marker=dict(color='red'), name='Anomalías'
    )
    st.plotly_chart(fig_anom_w, use_container_width=True)

else:  # Mes
    daily_m = df.assign(dia=df['fecha'].dt.date)
    monthly_avg = (daily_m
                   .groupby(['nombre_mes','intervalo'])[['planificados','reales']]
                   .mean().reset_index())
    fig = px.line(
        monthly_avg, x='intervalo', y=['planificados','reales'],
        facet_col='nombre_mes', facet_col_wrap=3,
        title='📊 Curva horaria promedio diario por Mes',
        labels={'intervalo':'Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'red','reales':'blue'}
    )
    fig.update_layout(hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
