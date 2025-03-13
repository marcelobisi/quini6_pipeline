# test/test_preprocessing.py
# test/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np  # Añadir esta línea
from quini6_pipeline.data.preprocessing import limpiar_datos

def test_rango_numeros():
    data = {
        'n1': [1, 30, 3],
        'n2': [2, 2, 45],
        'n3': [3, 3, 3],
        'n4': [4, 4, 4],
        'n5': [5, 5, 5],
        'n6': [6, 6, 6],
        'fecha': ['2023-01-01', '2023-01-02', '2023-01-03']
    }
    df = pd.DataFrame(data)
    df_limpio = limpiar_datos(df)
    assert len(df_limpio) == 1  # Solo la fila con n1=1, n2=2, etc., debe permanecer

def test_limpieza_datos_basica():
    data = {
        'n1': [1, 30, 3],
        'n2': [2, 2, 45],
        'n3': [3, 3, 3],
        'n4': [4, 4, 4],
        'n5': [5, 5, 5],
        'n6': [6, 6, 6],
        'fecha': ['2023-01-01', '2023-01-02', '2023-01-03']
    }
    df = pd.DataFrame(data)
    df_limpio = limpiar_datos(df)
    assert len(df_limpio) == 1  # Debe eliminar las filas con valores inválidos

def test_split_temporal():
    import numpy as np  # Añadir esta línea
    
    fechas = pd.date_range('2023-01-01', periods=100)
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 6)
    
    # Asegurar que se usa la función split_data correcta
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, validation_size=0.1)
    
    assert len(X_train) == 70
    assert len(X_val) == 10
    assert len(X_test) == 20