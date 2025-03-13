# config.py
import os

# Rutas de archivos
DATASET_PATH = 'data/quini6_sorteos.csv'  # Ruta relativa dentro del proyecto
MODEL_SAVE_DIR = 'models/saved_models/'
LOG_FILE_PATH = 'logs/pipeline.log'

# Hiperparámetros del modelo
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 100
SEQUENCE_LENGTH = 10

# Configuración de la base de datos (EJEMPLO - ADAPTAR)
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'brunilda'
DB_NAME = 'quini6_predict'