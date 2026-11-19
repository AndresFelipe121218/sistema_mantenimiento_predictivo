
# Sistema de Mantenimiento Predictivo

Sistema de mantenimiento predictivo para un proceso de mecanizado mediante aprendizaje automático.

## Descripción

Proyecto académico que implementa un modelo de Machine Learning para identificar posibles fallas en un proceso de mecanizado a partir de variables operativas.

## Funcionalidades

- Ingreso de variables operativas.
- Predicción de posibles fallas.
- Visualización del resultado y probabilidad estimada.
- Presentación de métricas de evaluación del modelo.

## Tecnologías

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## Modelo seleccionado

Máquina de soporte vectorial (SVM).

Métricas de evaluación:

- Accuracy: 91.75 %
- Precision: 27.85 %
- Recall: 89.71 %
- F1-score: 42.51 %

El modelo prioriza la detección de fallas, aunque puede generar falsas alarmas.

## Ejecución del proyecto

1. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecutar la aplicación:

```bash
streamlit run app.py
```
