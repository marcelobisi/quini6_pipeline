# persistence/model_saver.py
import torch
import io
from quini6_pipeline.persistence import database_manager # Importar db_manager

def save_model_to_db(modelo, model_name, description, metrics_json):
    buffer = io.BytesIO() # Buffer en memoria para serializar el modelo
    torch.save(modelo.state_dict(), buffer) # Guardar solo el state_dict (pesos)
    model_data_blob = buffer.getvalue() # Obtener los datos como bytes

    query = "CALL InsertModelo(:nombre, :descripcion, :datos_modelo, :metricas)" # Usar procedimiento almacenado
    params = {
        'nombre': model_name,
        'descripcion': description,
        'datos_modelo': model_data_blob,
        'metricas': metrics_json # Asume que metrics_json ya es un JSON serializable
    }
    database_manager.execute_query(query, params)
    print(f"Modelo '{model_name}' guardado en la base de datos.")

def load_model_from_db(model_name, model_class): # model_class es la clase del modelo (ej: ModeloPredictorModalidad)
    query = "SELECT datos_modelo FROM modelos WHERE nombre = :nombre"
    params = {'nombre': model_name}
    result = database_manager.fetch_data(query, params)
    if result and result[0] and result[0][0]:
        model_data_blob = result[0][0] # Obtener el BLOB de datos del modelo
        buffer = io.BytesIO(model_data_blob) # Buffer para deserializar
        model_state_dict = torch.load(buffer) # Cargar state_dict
        modelo = model_class(...) # Crear instancia del modelo (necesitarás saber los argumentos de __init__)
        modelo.load_state_dict(model_state_dict) # Cargar pesos en el modelo
        print(f"Modelo '{model_name}' cargado desde la base de datos.")
        return modelo
    else:
        print(f"Modelo '{model_name}' no encontrado en la base de datos.")
        return None