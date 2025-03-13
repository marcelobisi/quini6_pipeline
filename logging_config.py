# logging_config.py
import logging
import logging.config
from .persistence import database_manager # Import Relativo (Correcto)
from quini6_pipeline import config # Importar config para configuraciones DB

class DatabaseHandler(logging.Handler):
    """
    Handler de logging personalizado que envía los registros de log a la base de datos.
    """
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)

    def emit(self, record):
        """
        Emite un registro de log. Este método es llamado automáticamente por el sistema de logging.
        """
        log_entry = self.format(record) # Formatear el registro usando el formatter asignado
        engine = database_manager.get_db_engine() # Obtener conexión a la BD en cada emisión (para manejar conexiones intermitentes)
        if engine:
            try:
                query = "INSERT INTO logs (timestamp, level, name, message) VALUES (:timestamp, :level, :name, :message)" # **Consulta con marcadores de posición *nombrados***
                log_data = [ # **log_data como LISTA de *DICCIONARIOS***
                    {  # Esto es un DICCIONARIO dentro de la lista
                        "timestamp": record.asctime, # Formatear timestamp
                        "level": record.levelname, # level
                        "name": record.name, # name
                        "message": log_entry  # Usar el mensaje ya formateado
                    }
                ] # log_data ahora es UNA LISTA que contiene UN DICCIONARIO
                print(f"Debugging log_data: {log_data}") # Debugging line - DEJA ESTA LÍNEA
                database_manager.execute_query(query, log_data) # Ejecutar consulta con parámetros
            except Exception as db_error:
                print(f"Error al insertar log en la base de datos: {db_error}") # Imprimir a consola en caso de error al loggear en BD
                # **Opcional:** También podrías loggear este error a un archivo de respaldo si quieres asegurar la persistencia de errores de logging.


def setup_logging(log_file_path='pipeline.log', level=logging.INFO, log_to_database=True): # Añadir parámetro log_to_database
    """
    Configura el sistema de logging.

    Args:
        log_file_path (str): Ruta del archivo para guardar logs (si se usa FileHandler).
        level (int): Nivel de logging (ej: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL).
        log_to_database (bool): Indica si se deben guardar los logs en la base de datos.
    """
    format_str = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    date_format_str = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_str, datefmt=date_format_str)

    logging.basicConfig(  # Configuración básica (ya no usamos FileHandler directamente aquí)
        level=level,
        format=format_str,
        datefmt=date_format_str,
        #handlers=[] # No configuramos handlers aquí directamente, los añadimos luego
    )

    # Handler para la consola (siempre útil para ver logs importantes rápidamente)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING) # Mostrar warnings y errores en la consola
    console_formatter = logging.Formatter('%(levelname)s - %(message)s') # Formato más simple para consola
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler) # Añadir handler a logger raíz

    if log_to_database: # Condicional para activar/desactivar logging a BD
        database_handler = DatabaseHandler(level=level) # Handler para la base de datos
        database_handler.setFormatter(formatter) # Usar el mismo formatter que para el archivo (o uno diferente)
        logging.getLogger('').addHandler(database_handler) # Añadir database handler al logger raíz
        logging.info("Logging a la base de datos activado.")


    logging.info(f"Logging configurado. Nivel: {logging.getLevelName(level)}, Log a archivo: {log_file_path if 'FileHandler' in [h.__class__.__name__ for h in logging.getLogger('').handlers] else 'DESACTIVADO'}, Log a BD: {'ACTIVADO' if log_to_database else 'DESACTIVADO'}") # Info actualizada sobre logging a BD

if __name__ == '__main__':
    setup_logging(log_to_database=True, level=logging.DEBUG) # Ejemplo de uso con logging a BD activado y nivel DEBUG
    logger_prueba = logging.getLogger("prueba_logger") # Logger específico para pruebas
    logger_prueba.debug("Mensaje de debug - debería ir a la BD.")
    logger_prueba.info("Mensaje de información - debería ir a la BD.")
    logger_prueba.warning("Mensaje de advertencia - debería ir a la consola y BD.")
    logger_prueba.error("Mensaje de error - debería ir a la consola y BD.")
    logger_prueba.critical("Mensaje crítico - debería ir a la consola y BD.")
    logging.warning("Mensaje de warning directo al logger raíz - debería ir a la consola y BD.") # Prueba logger raíz también
    print("Revisar la tabla 'logs' en la base de datos para ver los registros.")