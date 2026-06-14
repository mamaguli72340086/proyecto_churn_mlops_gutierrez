"""
API de predicción de churn con FastAPI.

La API carga un modelo serializado, valida los datos de entrada
y devuelve una predicción enriquecida con nivel de riesgo y recomendación.

MEJORA TÉCNICA PERSONAL (MLOps):
- Se agregan los campos `nivel_riesgo` y `recomendacion` a la respuesta de /predict.
- La respuesta es más interpretable y accionable para el negocio.

----------------------------------------
PASOS PARA EJECUTAR ESTA API:
----------------------------------------
1. Entrenar el modelo (primera vez o cuando se quiera actualizar):
   > python src/entrenar_modelo.py
   Esto generará el archivo models/modelo_churn_v1.joblib

2. Instalar dependencias (si no se hizo):
   > pip install -r requirements.txt

3. Ejecutar la API:
   > uvicorn main:app --reload
   (Asegúrate de estar en la carpeta donde está este main.py)

4. Probar con un cliente HTTP (ejemplo con curl):
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"antiguedad": 24, "cargo_mensual": 85.5, "reclamos": 2}'

5. Ver documentación interactiva:
   http://localhost:8000/docs
----------------------------------------
"""

from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ========================= CONFIGURACIÓN =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1.joblib"

VERSION_MODELO = "modelo_churn_v1"
AUTOR = "Marlene Martha Gutierrez Limachi"

if not MODEL_PATH.exists():
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

modelo = joblib.load(MODEL_PATH)

# ========================= ESQUEMAS DE DATOS =========================
class ClienteEntrada(BaseModel):
    antiguedad: int = Field(
        ...,
        ge=0,
        le=120,
        description="Antigüedad del cliente expresada en meses",
        examples=[12],
    )
    cargo_mensual: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Cargo mensual del cliente",
        examples=[95.5],
    )
    reclamos: int = Field(
        ...,
        ge=0,
        le=50,
        description="Cantidad de reclamos recientes",
        examples=[3],
    )

class PrediccionSalida(BaseModel):
    prediccion: str               # "alto_riesgo" o "bajo_riesgo"
    probabilidad: float
    nivel_riesgo: Literal["Bajo", "Medio", "Alto"]
    recomendacion: str
    version_modelo: str
    autor: str

# ========================= APLICACIÓN FASTAPI =========================
app = FastAPI(
    title="API de predicción de churn",
    description="Servicio académico ML-Ops para estimar riesgo de abandono.",
    version="1.0.0",
)

@app.get("/")
def inicio() -> dict[str, str]:
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": AUTOR,
        "fecha_entrenamiento": "13-06-2026",
        "version_modelo": VERSION_MODELO,
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "estado": "ok",
        "modelo": VERSION_MODELO,
    }

@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    try:
        # Preparar entrada para el modelo
        X = [[
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
        ]]

        # Probabilidad de churn (clase 1)
        probabilidad = float(modelo.predict_proba(X)[0][1])
        etiqueta = "alto_riesgo" if probabilidad >= 0.50 else "bajo_riesgo"

        # --- MEJORA TÉCNICA: nivel de riesgo y recomendación ---
        if probabilidad >= 0.70:
            nivel = "Alto"
            recomendacion = "Ofrecer descuento o campaña de retención inmediata"
        elif probabilidad >= 0.40:
            nivel = "Medio"
            recomendacion = "Enviar encuesta de satisfacción y monitorear uso"
        else:
            nivel = "Bajo"
            recomendacion = "Mantener comunicación estándar, sin acción especial"

        return PrediccionSalida(
            prediccion=etiqueta,
            probabilidad=round(probabilidad, 4),
            nivel_riesgo=nivel,
            recomendacion=recomendacion,
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from exc