"""
API predictiva de churn con monitoreo básico y respuesta enriquecida.

MEJORA TÉCNICA PERSONAL (MLOps):
- Se agregan los campos `nivel_riesgo` y `recomendacion` a la respuesta de /predict.
- La respuesta es más interpretable y accionable para el negocio.

MONITOREO (Guía 7):
- Logging a archivo y consola.
- Middleware para medir latencia y contar solicitudes/códigos HTTP.
- Detección de valores fuera del rango histórico (señal de drift).
- Contadores acumulados con protección de concurrencia.
- Endpoint /metrics para consultar el resumen.
"""

# ============================================================
# BLOQUE 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
import logging
import os
from collections import Counter
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

# ============================================================
# BLOQUE 2. CONFIGURACIÓN GENERAL
# ============================================================
# Ruta raíz del proyecto (asumiendo que este archivo está en api/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ruta del modelo serializado
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1.joblib"

# Carpeta de logs
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "monitor_api.log"

VERSION_MODELO = "modelo_churn_v1"
AUTOR = "Marlene Martha Gutierrez Limachi"  # Personalizar con nombre y apellido

# ============================================================
# BLOQUE 3. RANGOS HISTÓRICOS DE REFERENCIA
# ============================================================
RANGOS_HISTORICOS = {
    "antiguedad": (1, 72),
    "cargo_mensual": (20.0, 150.0),
    "reclamos": (0, 7),
}

# ============================================================
# BLOQUE 4. LOGGING A ARCHIVO Y CONSOLA
# ============================================================
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("api_churn")

# ============================================================
# BLOQUE 5. VERIFICACIÓN Y CARGA DEL MODELO
# ============================================================
if not MODEL_PATH.exists():
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src/entrenar_modelo.py"
    )

modelo = joblib.load(MODEL_PATH)
logger.info("Modelo cargado correctamente: %s", VERSION_MODELO)

# ============================================================
# BLOQUE 6. CONTADORES DE MÉTRICAS EN MEMORIA
# ============================================================
metricas = {
    "solicitudes_totales": 0,
    "errores_validacion": 0,
    "errores_internos": 0,
    "predicciones_validas": 0,
    "predicciones_alto_riesgo": 0,
    "predicciones_bajo_riesgo": 0,
    "solicitudes_con_anomalias": 0,
    "latencia_acumulada_ms": 0.0,
    "latencia_maxima_ms": 0.0,
    "codigos_http": Counter(),
}
metricas_lock = Lock()

# ============================================================
# BLOQUE 7. MODELOS DE DATOS (Pydantic)
# ============================================================
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
    prediccion: str          # "alto_riesgo" o "bajo_riesgo"
    probabilidad: float
    nivel_riesgo: Literal["Bajo", "Medio", "Alto"]
    recomendacion: str
    version_modelo: str
    autor: str
    alertas_datos: list[str]   # Se agrega para mostrar anomalías

# ============================================================
# BLOQUE 8. DETECCIÓN DE VALORES FUERA DEL RANGO HISTÓRICO
# ============================================================
def detectar_anomalias(datos: ClienteEntrada) -> list[str]:
    alertas = []
    valores = datos.model_dump()
    for variable, valor in valores.items():
        if variable in RANGOS_HISTORICOS:
            minimo, maximo = RANGOS_HISTORICOS[variable]
            if valor < minimo or valor > maximo:
                alertas.append(
                    f"{variable}={valor} fuera del rango histórico "
                    f"[{minimo}, {maximo}]"
                )
    return alertas

# ============================================================
# BLOQUE 9. RESUMEN DE MÉTRICAS
# ============================================================
def resumen_metricas() -> dict:
    with metricas_lock:
        total = metricas["solicitudes_totales"]
        latencia_promedio = (
            metricas["latencia_acumulada_ms"] / total if total else 0.0
        )
        return {
            "version_modelo": VERSION_MODELO,
            "autor": AUTOR,
            "solicitudes_totales": total,
            "errores_validacion": metricas["errores_validacion"],
            "errores_internos": metricas["errores_internos"],
            "predicciones_validas": metricas["predicciones_validas"],
            "predicciones_alto_riesgo": metricas["predicciones_alto_riesgo"],
            "predicciones_bajo_riesgo": metricas["predicciones_bajo_riesgo"],
            "solicitudes_con_anomalias": metricas["solicitudes_con_anomalias"],
            "latencia_promedio_ms": round(latencia_promedio, 3),
            "latencia_maxima_ms": round(metricas["latencia_maxima_ms"], 3),
            "codigos_http": dict(metricas["codigos_http"]),
        }

# ============================================================
# BLOQUE 10. APLICACIÓN FASTAPI
# ============================================================
app = FastAPI(
    title="API de predicción de churn con monitoreo básico",
    description="Servicio académico ML-Ops con métricas, logs y respuesta enriquecida.",
    version="2.1.0",
)

# ============================================================
# BLOQUE 11. MIDDLEWARE PARA MEDIR LATENCIA Y CONTAR SOLICITUDES
# ============================================================
@app.middleware("http")
async def registrar_solicitud(request: Request, call_next):
    inicio = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        with metricas_lock:
            metricas["errores_internos"] += 1
        logger.exception("Error interno no controlado en %s", request.url.path)
        raise

    latencia_ms = (perf_counter() - inicio) * 1000

    with metricas_lock:
        metricas["solicitudes_totales"] += 1
        metricas["latencia_acumulada_ms"] += latencia_ms
        metricas["latencia_maxima_ms"] = max(
            metricas["latencia_maxima_ms"], latencia_ms
        )
        metricas["codigos_http"][str(response.status_code)] += 1

    logger.info(
        "Solicitud | metodo=%s | ruta=%s | estado=%s | latencia_ms=%.3f",
        request.method,
        request.url.path,
        response.status_code,
        latencia_ms,
    )

    response.headers["X-Process-Time-ms"] = f"{latencia_ms:.3f}"
    return response

# ============================================================
# BLOQUE 12. MANEJO DE ERRORES DE VALIDACIÓN
# ============================================================
@app.exception_handler(RequestValidationError)
async def registrar_error_validacion(request: Request, exc: RequestValidationError):
    with metricas_lock:
        metricas["errores_validacion"] += 1
    logger.warning(
        "Error de validación | ruta=%s | detalle=%s",
        request.url.path,
        exc.errors(),
    )
    return await request_validation_exception_handler(request, exc)

# ============================================================
# BLOQUE 13. ENDPOINTS BÁSICOS
# ============================================================
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
        "monitoreo": "activo",
    }

# ============================================================
# BLOQUE 14. ENDPOINT /metrics (MONITOREO)
# ============================================================
@app.get("/metrics")
def metrics() -> dict:
    """Devuelve el resumen acumulado de métricas."""
    return resumen_metricas()

# ============================================================
# BLOQUE 15. ENDPOINT /predict (CON MEJORA PERSONALIZADA)
# ============================================================
@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    try:
        # Detectar anomalías (rangos históricos)
        alertas = detectar_anomalias(datos)

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

        # Actualizar métricas
        with metricas_lock:
            metricas["predicciones_validas"] += 1
            metricas[f"predicciones_{etiqueta}"] += 1
            if alertas:
                metricas["solicitudes_con_anomalias"] += 1

        # Log de advertencia si hay anomalías
        if alertas:
            logger.warning(
                "Valores fuera de rango histórico: %s",
                alertas,
            )

        logger.info(
            "Predicción | resultado=%s | probabilidad=%.4f | alertas=%s",
            etiqueta,
            probabilidad,
            len(alertas),
        )

        # Respuesta enriquecida con alertas
        return PrediccionSalida(
            prediccion=etiqueta,
            probabilidad=round(probabilidad, 4),
            nivel_riesgo=nivel,
            recomendacion=recomendacion,
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
            alertas_datos=alertas,
        )

    except Exception as exc:
        with metricas_lock:
            metricas["errores_internos"] += 1
        logger.exception("No fue posible generar la predicción")
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from exc