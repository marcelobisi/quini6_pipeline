# pipeline.py
import logging
from quini6_pipeline import config, logging_config # Importar config y logging
from quini6_pipeline.data import data_loader, preprocessing
from quini6_pipeline.models import model_architectures
from quini6_pipeline.training import trainer, metrics
from quini6_pipeline.evaluation import evaluator
from quini6_pipeline.prediction import predictor
from quini6_pipeline.persistence import model_saver, scaler_saver, database_manager

logging_config.setup_logging(log_file_path=config.LOG_FILE_PATH) # Configurar logging al inicio

def run_pipeline():
    logging.info("Inicio del pipeline.")

    # 1. Cargar datos
    logging.info("Cargando datos...")
    df = data_loader.load_data_from_db() # O load_data_from_csv()
    if df is None:
        logging.error("Error al cargar datos. Pipeline abortado.")
        return

    # 2. Preprocesar datos
    logging.info("Preprocesando datos...")
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, scaler = preprocessing.preprocess_pipeline(df, config.SEQUENCE_LENGTH)
    # Guardar scaler en la base de datos (ej: scaler_saver.save_scaler_to_db(scaler, "nombre_scaler"))

    # 3. Definir modelo
    logging.info("Definiendo modelo...")
    input_size = # ... (Calcula input_size según config. y datos) ...
    output_size = # ... (Calcula output_size) ...
    modelo_global = model_architectures.ModeloPredictorModalidad(input_size, config.HIDDEN_SIZE, output_size)

    # 4. Entrenar modelo
    logging.info("Entrenando modelo...")
    historia_perdida_entrenamiento, historia_perdida_validacion = trainer.entrenar_modelo_global(
        modelo_global, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
    )
    # Guardar modelo entrenado en la base de datos (ej: model_saver.save_model_to_db(modelo_global, "ModeloGlobal_V1", "Descripción modelo global", metrics_json))

    # 5. Evaluar modelo
    logging.info("Evaluando modelo...")
    perdida_prueba, mae_prueba = evaluator.evaluar_modelo(modelo_global, X_test_tensor, y_test_tensor)
    logging.info(f"Pérdida en prueba: MSE={perdida_prueba:.4f}, MAE={mae_prueba:.4f}")
    # Guardar métricas de evaluación en la base de datos (ej: database_manager.execute_query("CALL UpdateModeloMetricas...", params))

    # 6. Generar predicciones (ejemplo: para el conjunto de prueba)
    logging.info("Generando predicciones...")
    # ... (Obtener una secuencia de entrada para predicción) ...
    prediccion_nuevo_sorteo = predictor.predecir_sorteo(modelo_global, X_test_tensor[0, :].numpy()) # Ejemplo con una secuencia de prueba

    # 7. Guardar predicciones (en base de datos o archivos)
    logging.info("Guardando predicciones...")
    # ... (Guardar predicciones y asociarlas al modelo usado en la base de datos, usando prediction_saver.py) ...

    logging.info("Pipeline completado exitosamente.")

if __name__ == "__main__":
    run_pipeline()