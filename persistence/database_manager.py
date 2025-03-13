# persistence/database_manager.py
# persistence/database_manager.py (continuación)
from typing import Optional, Dict, List, Union
import json
import logging
from datetime import date
from sqlalchemy import create_engine, text
from quini6_pipeline import config # Importar configuraciones

def get_db_engine():
    engine = create_engine(f'mariadb+mariadbconnector://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}/{config.DB_NAME}')
    return engine

def execute_query(query_string, params=None): # Función general para ejecutar consultas
    engine = get_db_engine()
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query_string), params) # **PASAR PARAMS DIRECTAMENTE (TUPLE O None)**
            connection.commit() # Si es INSERT, UPDATE, DELETE
            return result
    except Exception as e:
        print(f"Error al ejecutar consulta: {e}")
        return None

def fetch_data(query_string, params=None): # Función general para SELECT queries
    engine = get_db_engine()
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query_string), params)
            return result.fetchall() # Devuelve lista de tuplas (filas)
    except Exception as e:
        print(f"Error al ejecutar consulta de fetch: {e}")
        return None

# ... (Funciones más específicas usando execute_query y fetch_data para interactuar con las tablas,
#      ej:  insert_prediccion_db(), get_model_by_name(), save_scaler_to_db(), etc.) ...



# Configurar logging
logger = logging.getLogger(__name__)

# ======================== FUNCIONES PARA PREDICCIONES ========================
def insert_prediccion_db(
    sorteo: int,
    fecha: date,
    modalidad: str,
    numeros: List[int],
    probabilidades: List[float],
    prob_acumulada: float,
    nombre_modelo: str
) -> bool:
    """
    Inserta una predicción en la base de datos usando el procedimiento almacenado InsertPrediccion.
    
    Args:
        sorteo: Número de sorteo
        fecha: Fecha del sorteo
        modalidad: Una de ['TRADICIONAL', 'SEGUNDA', 'REVANCHA', 'SIEMPRE SALE']
        numeros: Lista de 6 números predichos
        probabilidades: Lista de 6 probabilidades correspondientes
        prob_acumulada: Probabilidad acumulada total
        nombre_modelo: Nombre del modelo usado para la predicción
    
    Returns:
        bool: True si la inserción fue exitosa, False en caso contrario
    """
    # Validación de ENUM
    modalidad = modalidad.upper()
    if modalidad not in {'TRADICIONAL', 'SEGUNDA', 'REVANCHA', 'SIEMPRE SALE'}:
        logger.error(f"Modalidad inválida: {modalidad}")
        return False
    
    # Construir parámetros para el stored procedure
    params = (
        sorteo,
        fecha,
        modalidad,
        *numeros,          # n1 a n6
        *probabilidades,    # prob1 a prob6
        prob_acumulada,
        nombre_modelo
    )
    
    try:
        # Llamar al stored procedure
        execute_query(
            "CALL InsertPrediccion(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            params
        )
        return True
    except Exception as e:
        logger.error(f"Error insertando predicción: {str(e)}")
        return False

# ======================== FUNCIONES PARA MODELOS ========================
def get_model_by_name(model_name: str) -> Optional[Dict]:
    """
    Obtiene un modelo por su nombre desde la base de datos.
    
    Returns:
        Dict con estructura: {
            'id': int,
            'nombre': str,
            'fecha_entrenamiento': datetime,
            'descripcion': str,
            'datos_modelo': bytes (serializado),
            'metricas': dict
        }
    """
    query = """
        SELECT id, nombre, fecha_entrenamiento, descripcion, datos_modelo, metricas
        FROM modelos
        WHERE nombre = %s
        ORDER BY fecha_entrenamiento DESC
        LIMIT 1
    """
    
    try:
        result = fetch_data(query, (model_name,))
        if result:
            row = result[0]
            return {
                'id': row[0],
                'nombre': row[1],
                'fecha_entrenamiento': row[2],
                'descripcion': row[3],
                'datos_modelo': row[4],
                'metricas': json.loads(row[5]) if row[5] else None
            }
        return None
    except Exception as e:
        logger.error(f"Error obteniendo modelo {model_name}: {str(e)}")
        return None

def insert_modelo_db(
    nombre: str,
    datos_modelo: bytes,
    metricas: Dict,
    descripcion: str = None
) -> bool:
    """
    Inserta un nuevo modelo en la base de datos usando el stored procedure InsertModelo.
    
    Args:
        nombre: Nombre único del modelo
        datos_modelo: Bytes del modelo serializado (pickle/joblib/etc)
        metricas: Diccionario con métricas de evaluación
        descripcion: Descripción opcional del modelo
    
    Returns:
        bool: True si la inserción fue exitosa
    """
    try:
        execute_query(
            "CALL InsertModelo(%s, %s, %s, %s)",
            (
                nombre,
                descripcion,
                datos_modelo,
                json.dumps(metricas)
        ))
        return True
    except Exception as e:
        logger.error(f"Error insertando modelo {nombre}: {str(e)}")
        return False

# ======================== FUNCIONES PARA SCALERS ========================
def save_scaler_to_db(nombre: str, scaler_data: Dict) -> bool:
    """
    Guarda un scaler en la tabla scalers_json.
    
    Args:
        nombre: Nombre único del scaler
        scaler_data: Datos del scaler en formato serializable a JSON
    
    Returns:
        bool: True si la operación fue exitosa
    """
    try:
        execute_query(
            "INSERT INTO scalers_json (nombre, datos) VALUES (%s, %s)"
            " ON DUPLICATE KEY UPDATE datos = VALUES(datos)",
            (nombre, json.dumps(scaler_data)))
        return True
    except Exception as e:
        logger.error(f"Error guardando scaler {nombre}: {str(e)}")
        return False

def get_scaler_from_db(nombre: str) -> Optional[Dict]:
    """
    Recupera un scaler de la base de datos.
    
    Returns:
        Dict con los datos del scaler o None si no existe
    """
    try:
        result = fetch_data(
            "SELECT datos FROM scalers_json WHERE nombre = %s",
            (nombre,))
        return json.loads(result[0][0]) if result else None
    except Exception as e:
        logger.error(f"Error obteniendo scaler {nombre}: {str(e)}")
        return None

# ======================== FUNCIONES ADICIONALES ========================
def get_evaluaciones_por_modelo(model_name: str) -> List[Dict]:
    """Ejecuta el stored procedure GetEvaluacionesPorModelo"""
    try:
        result = fetch_data("CALL GetEvaluacionesPorModelo(%s)", (model_name,))
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Error obteniendo evaluaciones para {model_name}: {str(e)}")
        return []

def get_predicciones_por_modelo(model_name: str) -> List[Dict]:
    """Ejecuta el stored procedure GetPrediccionesPorModelo"""
    try:
        result = fetch_data("CALL GetPrediccionesPorModelo(%s)", (model_name,))
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Error obteniendo predicciones para {model_name}: {str(e)}")
        return []

def update_model_metrics(model_id: int, new_metrics: Dict) -> bool:
    """Actualiza las métricas de un modelo usando el stored procedure UpdateModeloMetricas"""
    try:
        execute_query(
            "CALL UpdateModeloMetricas(%s, %s)",
            (model_id, json.dumps(new_metrics)))
        return True
    except Exception as e:
        logger.error(f"Error actualizando métricas del modelo {model_id}: {str(e)}")
        return False
