# data/data_loader.py
import pandas as pd
from sqlalchemy import create_engine
from quini6_pipeline import config # Importar configuraciones

def load_data_from_db(): # Ejemplo con SQLAlchemy (ADAPTAR CONECTOR/ORM)
    engine = create_engine(f'mariadb+mariadbconnector://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}/{config.DB_NAME}')
    try:
        with engine.connect() as connection:
            query = "SELECT * FROM sorteos ORDER BY fecha ASC" # Ejemplo de consulta
            df = pd.read_sql_query(query, connection)
            print("Datos cargados desde la base de datos.")
            return df
    except Exception as e:
        print(f"Error al cargar datos desde la base de datos: {e}")
        return None