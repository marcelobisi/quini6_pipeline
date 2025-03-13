'''
Próximos Pasos y Mejoras:

Implementar la Inserción Real en actualizar_base_de_datos(): Reemplaza el placeholder con la lógica real para insertar los nuevos_sorteos_df en la tabla sorteos (usando pandas.to_sql, sentencias SQL INSERT manuales o procedimientos almacenados).
Manejo de Errores Robusto en actualizar_base_de_datos(): Asegúrate de agregar un manejo de errores completo en la función de inserción real para controlar posibles problemas al escribir en la base de datos (ej: violación de claves únicas, errores de tipo de datos, problemas de conexión). Considera usar bloques try...except y registrar errores en el log.
Pruebas Unitarias (Recomendado): Escribe pruebas unitarias para las funciones en data_ingestion.py para verificar que funcionan correctamente (especialmente conectar_db(), cargar_sorteos_db(), verificar_nuevos_sorteos()). Las pruebas unitarias te ayudarán a asegurar la calidad y la fiabilidad de tu código.
Frecuencia de Ejecución: Decide con qué frecuencia necesitas ejecutar este pipeline de ingesta de datos para mantener tu base de datos actualizada con los últimos sorteos del Quini 6. Podrías programar la ejecución del script data_ingestion.py con un programador de tareas del sistema operativo (ej: cron en Linux/macOS, "Programador de tareas" en Windows) para que se ejecute automáticamente a intervalos regulares (ej: diariamente, semanalmente, o antes de cada entrenamiento del modelo).
Validación de Datos Ingresados: Considera agregar validaciones adicionales a los datos que se están ingiriendo (ej: verificar que los números estén dentro del rango 00-45, que la modalidad sea una de las válidas, etc.) para asegurar la calidad de los datos en la base de datos.
'''
# quini6_pipeline/data/data_ingestion.py
import pandas as pd
from sqlalchemy import create_engine, text
from quini6_pipeline import config  # Importar configuraciones desde config.py
import logging

logger = logging.getLogger(__name__) # Logger para este módulo

def conectar_db():
    """
    Establece una conexión a la base de datos MariaDB utilizando SQLAlchemy.

    Retorna:
        sqlalchemy.engine.Engine: Objeto Engine de SQLAlchemy si la conexión es exitosa,
                                    None en caso de error.
    """
    try:
        engine = create_engine(f'mariadb+mariadbconnector://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}/{config.DB_NAME}')
        engine.connect() # Intenta conectar para verificar la conexión
        logger.info("Conexión a la base de datos establecida exitosamente.")
        return engine
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        return None

def cargar_sorteos_db(engine):
    """
    Carga todos los sorteos desde la tabla 'sorteos' de la base de datos.

    Args:
        engine (sqlalchemy.engine.Engine): Objeto Engine de SQLAlchemy para la conexión a la base de datos.

    Retorna:
        pandas.DataFrame: DataFrame de Pandas con los datos de los sorteos,
                          None en caso de error.
    """
    try:
        query = "SELECT * FROM sorteos ORDER BY fecha ASC, sorteo ASC, modalidad ASC"
        df_sorteos = pd.read_sql_query(query, engine)
        logger.info(f"Cargados {len(df_sorteos)} registros de sorteos desde la base de datos.")
        return df_sorteos
    except Exception as e:
        logger.error(f"Error al cargar sorteos desde la base de datos: {e}")
        return None

def verificar_nuevos_sorteos(engine, df_sorteos_actual):
    """
    Verifica si existen nuevos sorteos en la base de datos que no estén en el DataFrame actual.

    Args:
        engine (sqlalchemy.engine.Engine): Objeto Engine de SQLAlchemy para la conexión a la base de datos.
        df_sorteos_actual (pandas.DataFrame): DataFrame actual con los sorteos conocidos.

    Retorna:
        pandas.DataFrame: DataFrame con los nuevos sorteos encontrados,
                          DataFrame vacío si no hay nuevos sorteos,
                          None en caso de error o si no se puede obtener el último sorteo de la BD.
    """
    try:
        # Obtener el último sorteo del DataFrame actual (si existe)
        if not df_sorteos_actual.empty:
            ultimo_sorteo_df = df_sorteos_actual.sort_values(by=['fecha', 'sorteo'], ascending=False).iloc[0]
            fecha_ultimo_sorteo_actual = ultimo_sorteo_df['fecha']
            numero_ultimo_sorteo_actual = ultimo_sorteo_df['sorteo']
        else:
            fecha_ultimo_sorteo_actual = None
            numero_ultimo_sorteo_actual = None

        # Consulta para obtener sorteos más recientes que el último sorteo actual
        query = """
            SELECT * FROM sorteos
            WHERE (fecha > :fecha_ultimo) OR (fecha = :fecha_ultimo AND sorteo > :sorteo_ultimo)
            ORDER BY fecha ASC, sorteo ASC, modalidad ASC
        """
        params = {
            'fecha_ultimo': fecha_ultimo_sorteo_actual if fecha_ultimo_sorteo_actual else '1900-01-01', # Fecha antigua si no hay sorteos actuales
            'sorteo_ultimo': numero_ultimo_sorteo_actual if numero_ultimo_sorteo_actual else 0
        }

        df_nuevos_sorteos = pd.read_sql_query(sql=text(query), con=engine, params=params)

        if not df_nuevos_sorteos.empty:
            logger.info(f"Encontrados {len(df_nuevos_sorteos)} nuevos sorteos en la base de datos.")
        else:
            logger.info("No se encontraron nuevos sorteos en la base de datos.")

        return df_nuevos_sorteos

    except Exception as e:
        logger.error(f"Error al verificar nuevos sorteos: {e}")
        return None

def actualizar_base_de_datos(engine, nuevos_sorteos_df):
    """
    Actualiza la base de datos con un DataFrame de nuevos sorteos.
    En este ejemplo, simplemente imprime los nuevos sorteos encontrados.
    En un escenario real, aquí insertarías los nuevos sorteos en la tabla 'sorteos'.

    Args:
        engine (sqlalchemy.engine.Engine): Objeto Engine de SQLAlchemy para la conexión a la base de datos.
        nuevos_sorteos_df (pandas.DataFrame): DataFrame con los nuevos sorteos a actualizar.
    """
    if nuevos_sorteos_df is None or nuevos_sorteos_df.empty:
        logger.info("No hay nuevos sorteos para actualizar la base de datos.")
        return

    try:
        # *** Lógica para INSERTAR los nuevos_sorteos_df en la tabla 'sorteos' ***
        # *** Este es un placeholder. Debes implementar la inserción real usando SQL INSERT o pandas to_sql ***
        # *** Ejemplo PLACEHOLDER (solo imprime los nuevos sorteos):
        print("\n--- Nuevos Sorteos Encontrados (Placeholder - Implementar Inserción Real) ---")
        print(nuevos_sorteos_df)
        print("--- Fin Nuevos Sorteos ---")
        logger.warning("Actualización de base de datos PLACEHOLDER - Implementar inserción real a la BD.")


        logger.info(f"Base de datos actualizada con {len(nuevos_sorteos_df)} nuevos sorteos. (PLACEHOLDER)")

    except Exception as e:
        logger.error(f"Error al actualizar la base de datos: {e}")


def ingest_data_pipeline():
    """
    Función principal que orquesta el pipeline de ingesta de datos:
    1. Conecta a la base de datos.
    2. Carga los sorteos existentes.
    3. Verifica si hay nuevos sorteos.
    4. Actualiza la base de datos con los nuevos sorteos (PLACEHOLDER - Implementar).

    Retorna:
        pandas.DataFrame: DataFrame con todos los sorteos (existentes y nuevos),
                          None en caso de error en la ingesta.
    """
    logger.info("Iniciando pipeline de ingesta de datos.")

    engine = conectar_db()
    if engine is None:
        logger.error("Ingesta de datos fallida: No se pudo conectar a la base de datos.")
        return None

    df_sorteos_actual = cargar_sorteos_db(engine)
    if df_sorteos_actual is None:
        logger.error("Ingesta de datos fallida: No se pudieron cargar los sorteos iniciales.")
        return None

    df_nuevos_sorteos = verificar_nuevos_sorteos(engine, df_sorteos_actual)
    if df_nuevos_sorteos is None:
        logger.warning("No se pudieron verificar nuevos sorteos, continuando con los datos existentes.")
        # Continuar con df_sorteos_actual, incluso si hubo un problema al verificar nuevos.
    elif not df_nuevos_sorteos.empty:
        actualizar_base_de_datos(engine, df_nuevos_sorteos) # PLACEHOLDER - Implementar inserción real

        # Concatenar los nuevos sorteos al DataFrame actual (si se actualizaron en la BD en la función real)
        # df_sorteos_actual = pd.concat([df_sorteos_actual, df_nuevos_sorteos], ignore_index=True) # Descomentar en la implementación real

    logger.info("Pipeline de ingesta de datos completado.")
    return df_sorteos_actual # Retorna el DataFrame (actualizado o no)

if __name__ == "__main__":
    from quini6_pipeline import logging_config # Importar logging_config para ejecutar standalone

    logging_config.setup_logging() # Configurar logging para ejecución standalone

    df_sorteos = ingest_data_pipeline()

    if df_sorteos is not None:
        print("\n--- Primeras filas del DataFrame de Sorteos ---")
        print(df_sorteos.head())
        print("\n--- Últimas filas del DataFrame de Sorteos ---")
        print(df_sorteos.tail())
        print(f"\nTotal de sorteos en el DataFrame: {len(df_sorteos)}")
    else:
        print("\nError durante la ingesta de datos. Revisar logs para más detalles.")