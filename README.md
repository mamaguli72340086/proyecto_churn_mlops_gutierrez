# objetivo del Modulo
-Diseñar e implementar un flujo básico de ML-Ops para la puesta en producción de modelos de Machine Learning, integrando prácticas de gestión del ciclo de vida, reproducibilidad, control de versiones, automatización, construcción de APIs, contenedorización, pruebas funcionales, monitoreo y documentación técnica, con el propósito de asegurar que los modelos puedan operar de forma confiable, trazable, escalable y mantenible en entornos productivos o preproductivos.

# Proyecto Churn MLOps

Este proyecto corresponde a una práctica inicial del módulo de MLOps.

El objetivo es construir una estructura básica de trabajo para un proyecto de Machine Learning que permita:

- Preparar datos.
- Entrenar un modelo.
- Evaluar métricas.
- Guardar el modelo entrenado.
- Exponer el modelo mediante una API.
- Ejecutar pruebas básicas.

## Problema del proyecto

Se trabajará con un caso simplificado de predicción de abandono de clientes, conocido como churn.

El modelo intentará predecir si un cliente podría abandonar un servicio, utilizando variables como edad, antigüedad, saldo promedio, reclamos y uso de aplicación móvil.

## Estructura del proyecto

```text
proyecto_churn_mlops
├── data
├── notebooks
├── src
├── models
├── api
├── tests
├── docs
├── README.md
└── requirements.txt
```

## Carpetas principales

- `data`: contiene los datos del proyecto.
- `notebooks`: contiene análisis exploratorios.
- `src`: contiene los scripts principales del modelo.
- `models`: contiene el modelo entrenado.
- `api`: contiene la API del modelo.
- `tests`: contiene pruebas automáticas.
- `docs`: contiene documentación y métricas.

## Flujo inicial del proyecto

El flujo básico será:

1. Preparar los datos.
2. Entrenar el modelo.
3. Evaluar el modelo.
4. Guardar las métricas.
5. Crear una API básica.
6. Probar el funcionamiento inicial.
## Control de versiones

Este proyecto utiliza Git para registrar cambios y GitHub para respaldar el repositorio en la nube.

## Autores

- marlenemarthagutierrezlimachi@gmail.com 
- https://github.com/mamaguli72340086/

## 🏗️ Arquitectura

![Arquitectura MLOps](https://via.placeholder.com/800x400?text=Diagrama+MLOps)

1. **DVC** – versionado de datasets y pipelines de datos.
2. **MLflow** – registro de parámetros, métricas y modelos.
3. **GitHub Actions** – CI/CD: test, lint, y despliegue condicional.
4. **FastAPI + Docker** – servidor de inferencia.
5. **Evidently** – monitoreo de calidad de datos y drift.

## Requisitos

- Python 3.10
- Docker (opcional)
- DVC (>2.0)
- MLflow (>1.30)
- Cuenta en GitHub (con Actions habilitadas)

## Instalacion

git clone https://github.com/mamaguli72340086/proyecto_churn_mlops-1
cd churn_mlops
make setup  # o manual: python -m venv .venv; source .venv/bin/activate; pip install -r requirements.txt