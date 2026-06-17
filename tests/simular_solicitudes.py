"""
Simulación básica de solicitudes para observar el monitoreo y la mejora técnica.

Este script permite generar tráfico controlado hacia la API predictiva
de churn desarrollada durante el laboratorio.

Objetivos:
1. Enviar solicitudes válidas al endpoint POST /predict.
2. Enviar una solicitud atípica para generar alertas de datos.
3. Enviar una solicitud inválida para comprobar el error HTTP 422.
4. Mostrar la latencia informada por el middleware de la API.
5. Consultar el resumen acumulado mediante el endpoint GET /metrics.
6. Validar la MEJORA TÉCNICA: verificar los campos `nivel_riesgo`, 
   `recomendacion` y `alertas_datos` en la respuesta.

Importante:
- La API debe estar activa antes de ejecutar este archivo.
- Este script no entrena el modelo, ni modifica la API.
- Solamente simula solicitudes para observar su comportamiento.
"""

# ============================================================
# BLOQUE 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
from pprint import pprint
import requests

# ============================================================
# BLOQUE 2. CONFIGURACIÓN GENERAL
# ============================================================
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 10

# ============================================================
# BLOQUE 3. CASOS DE PRUEBA (ACTUALIZADOS)
# ============================================================
CASOS = [
    {
        "nombre": "cliente_estable",
        "datos": {
            "antiguedad": 48,
            "cargo_mensual": 55.0,
            "reclamos": 0,
        },
        # Esperado: bajo_riesgo, nivel Bajo, sin alertas
    },
    {
        "nombre": "cliente_estable_extremo",
        "datos": {
            "antiguedad": 72,
            "cargo_mensual": 20.0,
            "reclamos": 0,
        },
        # Esperado: bajo_riesgo, nivel Bajo, dentro del histórico
    },
    {
        "nombre": "cliente_riesgo_medio",
        "datos": {
            "antiguedad": 18,
            "cargo_mensual": 110.0,
            "reclamos": 3,
        },
        # Esperado: alto_riesgo (prob > 0.5), nivel Medio (prob ~0.5-0.7)
    },
    {
        "nombre": "cliente_alto_riesgo",
        "datos": {
            "antiguedad": 4,
            "cargo_mensual": 145.0,
            "reclamos": 7,
        },
        # Esperado: alto_riesgo, nivel Alto, sin alertas (límite justo en 7 reclamos)
    },
    {
        "nombre": "cliente_atipico",
        "datos": {
            "antiguedad": 180,
            "cargo_mensual": 600.0,
            "reclamos": 35,
        },
        # Esperado: técnicamente válido, pero alertas_datos lleno (fuera de rango)
    },
    {
        "nombre": "cliente_invalido",
        "datos": {
            "antiguedad": 12,
            "cargo_mensual": -50.0,
            "reclamos": 1,
        },
        # Esperado: HTTP 422 (cargo_mensual negativo)
    },
]

# ============================================================
# BLOQUE 4. FUNCIÓN PARA MOSTRAR LA RESPUESTA (MEJORADA)
# ============================================================
def mostrar_respuesta(nombre: str, respuesta: requests.Response) -> None:
    """
    Presenta de forma ordenada el resultado de una solicitud.
    Ahora resalta explícitamente los campos de la MEJORA TÉCNICA.
    """
    print("\n" + "=" * 70)
    print(f"Caso: {nombre}")
    print(f"Estado HTTP: {respuesta.status_code}")

    # Latencia (monitoreo)
    latencia = respuesta.headers.get("X-Process-Time-ms")
    if latencia is not None:
        print(f"Latencia informada por API: {latencia} ms")
    else:
        print("Latencia informada por API: no disponible")

    # Intentar parsear JSON
    try:
        data = respuesta.json()
        print("\n--- Cuerpo de la respuesta (completo) ---")
        pprint(data)

        # --- SECCIÓN DESTACADA: MEJORA TÉCNICA ---
        if respuesta.status_code == 200:
            print("\n--- MEJORA TÉCNICA (nivel_riesgo / recomendacion) ---")
            print(f"  Nivel de riesgo: {data.get('nivel_riesgo', 'No disponible')}")
            print(f"  Recomendación: {data.get('recomendacion', 'No disponible')}")
            alertas = data.get('alertas_datos', [])
            if alertas:
                print(f"  Alertas de datos: {alertas}")
            else:
                print("  Alertas de datos: Ninguna (dentro de rango histórico)")
            print("-----------------------------------------------------")

    except requests.exceptions.JSONDecodeError:
        print("La respuesta no contiene un JSON válido.")
        print(respuesta.text)

# ============================================================
# BLOQUE 5. FUNCIÓN PARA ENVIAR UNA SOLICITUD A POST /predict
# ============================================================
def enviar_caso(caso: dict) -> None:
    nombre = caso["nombre"]
    datos = caso["datos"]
    try:
        respuesta = requests.post(
            f"{BASE_URL}/predict",
            json=datos,
            timeout=TIMEOUT,
        )
        mostrar_respuesta(nombre, respuesta)
    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print("Error: no fue posible conectarse con la API.")
        print("Verifique que Uvicorn se encuentre activo en otra terminal.")
    except requests.exceptions.Timeout:
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print(f"Error: la API no respondió en menos de {TIMEOUT} segundos.")
    except requests.exceptions.RequestException as exc:
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print(f"Error inesperado durante la solicitud: {exc}")

# ============================================================
# BLOQUE 6. FUNCIÓN PARA CONSULTAR GET /metrics
# ============================================================
def consultar_metricas() -> None:
    print("\n" + "=" * 70)
    print("RESUMEN ACUMULADO DE MÉTRICAS (MONITOREO)")
    try:
        respuesta_metricas = requests.get(
            f"{BASE_URL}/metrics",
            timeout=TIMEOUT,
        )
        print(f"Estado HTTP: {respuesta_metricas.status_code}")
        print("--- Métricas acumuladas ---")
        pprint(respuesta_metricas.json())
    except requests.exceptions.ConnectionError:
        print("Error: no fue posible consultar las métricas.")
        print("Verifique que la API se encuentre activa.")
    except requests.exceptions.Timeout:
        print(f"Error: la API no respondió en menos de {TIMEOUT} segundos.")
    except requests.exceptions.JSONDecodeError:
        print("Error: la respuesta de /metrics no contiene un JSON válido.")
    except requests.exceptions.RequestException as exc:
        print(f"Error inesperado durante la consulta: {exc}")

# ============================================================
# BLOQUE 7. FUNCIÓN PRINCIPAL
# ============================================================
def main() -> None:
    print("=" * 70)
    print("SIMULACIÓN DE SOLICITUDES PARA LA API PREDICTIVA")
    print("(Validando monitoreo + mejora técnica: nivel_riesgo y recomendacion)")
    print("=" * 70)

    for caso in CASOS:
        enviar_caso(caso)

    consultar_metricas()

# ============================================================
# BLOQUE 8. PUNTO DE ENTRADA DEL PROGRAMA
# ============================================================
if __name__ == "__main__":
    main()