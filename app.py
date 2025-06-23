import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 Visualizador de Contactos por Día, Semana o Mes")

# Subida del archivo
file = st.file_uploader("📂 Carga tu archivo de históricos", type=["csv", "xlsx"])

if file:
    # Lectura y normalización
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        'fecha': 'fecha',
        'tramo': 'intervalo',
        'planif. contactos': 'planificados',
        'contactos': 'reales'
    })

    df['fecha'] = pd.to_datetime(df['fecha'])
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['nombre_mes'] = df['fecha'].dt.strftime('%B')
    df['semana'] = df['fecha'].dt.isocalendar().week
    df['dia_semana'] = df['fecha'].dt.day_name()

    # Vistas disponibles
    vista = st.selectbox("Selecciona la vista", ["Por día", "Por semana", "Por mes"])

    if vista == "Por día":
        df_dia = df.groupby(['fecha', 'dia_semana'])[['planificados', 'reales']].sum().reset_index()
        fig = px.line(df_dia, x='fecha', y=['planificados', 'reales'],
                      labels={'value': 'Volumen', 'fecha': 'Fecha', 'variable': 'Tipo'},
                      color_discrete_map={'planificados': 'orange', 'reales': 'blue'},
                      title="📅 Contactos diarios por día de la semana")
        fig.update_layout(hovermode="x unified")

    elif vista == "Por semana":
        df_semana = df.groupby(['año', 'mes', 'nombre_mes', 'semana'])[['planificados', 'reales']].sum().reset_index()
        df_semana['etiqueta'] = df_semana['nombre_mes'] + " - Sem " + df_semana['semana'].astype(str)
        fig = px.line(df_semana, x='etiqueta', y=['planificados', 'reales'],
                      labels={'value': 'Volumen', 'etiqueta': 'Semana', 'variable': 'Tipo'},
                      color_discrete_map={'planificados': 'orange', 'reales': 'blue'},
                      title="📆 Contactos por semana del año")
        fig.update_layout(xaxis_tickangle=-45)

    else:  # Por mes
        df_mes = df.groupby(['año', 'mes', 'nombre_mes'])[['planificados', 'reales']].sum().reset_index()
        df_mes['etiqueta'] = df_mes['año'].astype(str) + " - " + df_mes['nombre_mes']
        fig = px.line(df_mes, x='etiqueta', y=['planificados', 'reales'],
                      labels={'value': 'Volumen', 'etiqueta': 'Mes', 'variable': 'Tipo'},
                      color_discrete_map={'planificados': 'orange', 'reales': 'blue'},
                      title="📊 Contactos por mes")
        fig.update_layout(xaxis_tickangle=-45)

    # Mostrar gráfico
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
