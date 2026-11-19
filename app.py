
import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Mantenimiento Predictivo",
    page_icon="⚙️",
    layout="wide"
)

# ==========================================
# CARGAR MODELO
# ==========================================

RUTA_MODELO = "modelo_mantenimiento_predictivo.pkl"

@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)

# ==========================================
# INTERFAZ
# ==========================================

st.title("⚙️ Sistema de Mantenimiento Predictivo")
st.subheader("Proceso de mecanizado mediante aprendizaje automático")

st.write(
    "Ingrese las condiciones operativas para evaluar "
    "si existe una posible falla en el proceso."
)

if not os.path.exists(RUTA_MODELO):
    st.error("No se encontró el archivo del modelo SVM.")
    st.stop()

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"No fue posible cargar el modelo: {e}")
    st.stop()

st.divider()

st.subheader("📋 Parámetros de operación")

col1, col2 = st.columns(2)

with col1:
    tipo = st.selectbox(
        "Tipo de producto",
        ["L", "M", "H"],
        index=1,
        help="Variante de producto: L (baja), M (media), H (alta)."
    )

    temperatura_aire = st.number_input(
        "Temperatura del aire (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    temperatura_proceso = st.number_input(
        "Temperatura del proceso (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

with col2:
    velocidad = st.number_input(
        "Velocidad de rotación (rpm)",
        min_value=0,
        max_value=5000,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=0.1
    )

    desgaste = st.number_input(
        "Desgaste de herramienta (min)",
        min_value=0,
        max_value=500,
        value=180
    )

st.divider()

# ==========================================
# PREDICCIÓN
# ==========================================

if st.button("Evaluar condiciones", type="primary", use_container_width=True):

    nueva_observacion = pd.DataFrame([{
        "Type": tipo,
        "Air temperature [K]": temperatura_aire,
        "Process temperature [K]": temperatura_proceso,
        "Rotational speed [rpm]": velocidad,
        "Torque [Nm]": torque,
        "Tool wear [min]": desgaste
    }])

    try:
        prediccion = modelo.predict(nueva_observacion)[0]

        st.subheader("📊 Resultado de la evaluación")

        if int(prediccion) == 1:
            st.error("⚠️ POSIBLE FALLA DETECTADA")
            st.write(
                "El modelo identifica condiciones asociadas "
                "a una posible falla."
            )
        else:
            st.success("✅ NO SE IDENTIFICA FALLA")
            st.write(
                "El modelo clasifica las condiciones ingresadas "
                "como una observación sin falla."
            )

        # Mostrar probabilidad si el modelo la soporta
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(nueva_observacion)[0]
            clases = list(modelo.classes_)

            if 1 in clases:
                indice_falla = clases.index(1)
                probabilidad = probabilidades[indice_falla] * 100

                st.metric(
                    "Probabilidad estimada de falla",
                    f"{probabilidad:.2f}%"
                )

        st.caption(
            "Resultado generado por el modelo SVM. "
            "Es una predicción académica y no sustituye "
            "una inspección técnica."
        )

        with st.expander("Ver datos evaluados"):
            st.dataframe(nueva_observacion, use_container_width=True)

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
# ==========================================
# DESEMPEÑO DEL MODELO
# ==========================================

st.divider()

st.subheader("📈 Desempeño del modelo SVM")

st.write(
    "Métricas obtenidas durante la evaluación "
    "del modelo sobre el conjunto de prueba."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "91,75 %")
col2.metric("Precision", "27,85 %")
col3.metric("Recall", "89,71 %")
col4.metric("F1-Score", "42,51 %")

st.info(
    "El modelo alcanza un Recall del 89,71 %, lo que significa "
    "que identifica aproximadamente el 89,71 % de las fallas reales "
    "del conjunto de prueba. Su Precision del 27,85 % indica que "
    "también genera falsas alarmas, por lo que sus resultados deben "
    "ser validados mediante una inspección técnica."
)      

st.divider()

st.caption(
    "Proyecto académico | Sistema de Mantenimiento Predictivo "
    "mediante Aprendizaje Automático"
)