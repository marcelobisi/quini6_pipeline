'''
Recuerda adaptar la selección de features, la división de datos temporal,
y la creación de secuencias a las necesidades específicas de los modelos que vayas a utilizar
y a tu objetivo de predicción. ¡No dudes en preguntar si tienes más dudas o necesitas ayuda con
la implementación de la división temporal o con la integración en el pipeline!
'''
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from quini6_pipeline import data_ingestion, logging_config
import logging

logger = logging.getLogger(__name__)

def limpiar_datos(df):
    """
    Realiza limpieza básica de datos:
    - Elimina filas con valores nulos en columnas numéricas (si existen).
    - Asegura que las columnas numéricas tengan el tipo de dato correcto (numérico).
    - Corrige formatos de fecha si es necesario (aunque pandas debería manejarlo bien).

    Args:
        df (pandas.DataFrame): DataFrame de sorteos.

    Returns:
        pandas.DataFrame: DataFrame limpio.
    """
    logger.info("Iniciando limpieza de datos...")

    # **Eliminar Nulos (si los hubiera en columnas numéricas):**
    columnas_numericas_n = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    filas_con_nulos_antes = df[columnas_numericas_n].isnull().any(axis=1).sum()
    if filas_con_nulos_antes > 0:
        logger.warning(f"Se encontraron {filas_con_nulos_antes} filas con valores nulos en columnas de números. Eliminando.")
        df.dropna(subset=columnas_numericas_n, inplace=True) # Eliminar filas con NaN en n1-n6
    else:
        logger.info("No se encontraron valores nulos en las columnas de números.")

    # **Asegurar tipo numérico (ya debería ser numérico al leer de la BD, pero por si acaso):**
    for col in columnas_numericas_n:
        df[col] = pd.to_numeric(df[col], errors='coerce') # 'coerce' convierte no numéricos a NaN, que ya manejamos arriba
    filas_con_nan_tipo_dato = df[columnas_numericas_n].isnull().any(axis=1).sum() # Verificar si quedaron NaN después de coerce
    if filas_con_nan_tipo_dato > 0:
        logger.warning(f"Después de asegurar tipo numérico, {filas_con_nan_tipo_dato} filas contienen NaN. Eliminando.")
        df.dropna(subset=columnas_numericas_n, inplace=True)

    # **Formato de fecha (pandas debería manejar la columna 'fecha' como datetime):**
    try:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='raise') # 'raise' lanza error si no puede parsear
        logger.info("Columna 'fecha' verificada y en formato datetime.")
    except Exception as e:
        logger.error(f"Error al convertir columna 'fecha' a formato datetime: {e}")
        # Decidir si levantar excepción o manejar de otra forma si la fecha es crucial
        raise ValueError("Error en el formato de la columna 'fecha'. Revisar datos.") from e


    logger.info("Limpieza de datos completada.")
    return df

def calcular_frecuencia_numeros(df):
    """
    Calcula la frecuencia de aparición de cada número (0-45) en todas las columnas de números (n1-n6).

    Args:
        df (pandas.DataFrame): DataFrame de sorteos.

    Returns:
        pandas.DataFrame: DataFrame con columnas 'freq_n{numero}' para cada número del 0 al 45,
                          representando la frecuencia de ese número hasta ese sorteo (acumulativa).
    """
    logger.info("Calculando frecuencia de números...")
    columnas_numericas = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    df_frecuencias = df[columnas_numericas].copy() # Trabajar sobre copia para no modificar original

    for numero in range(0, 46): # Números del 0 al 45 en Quini 6
        nombre_columna_freq = f'freq_n{numero:02d}' # Ej: freq_n00, freq_n01, ..., freq_n45
        df_frecuencias[nombre_columna_freq] = (df_frecuencias[columnas_numericas] == numero).sum(axis=1)

    logger.info("Frecuencia de números calculada.")
    return df_frecuencias.drop(columns=columnas_numericas) # Eliminar columnas originales n1-n6 para retornar solo frecuencias


def calcular_intervalo_ultimo_aparecimiento(df):
    """
    Calcula el intervalo (en número de sorteos) desde la última vez que apareció cada número (0-45)
    antes de cada sorteo en el DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame de sorteos (ordenado cronológicamente por fecha y sorteo).

    Returns:
        pandas.DataFrame: DataFrame con columnas 'intervalo_n{numero}' para cada número del 0 al 45,
                          representando el intervalo desde la última aparición de ese número.
                          Si es la primera vez que aparece, el intervalo será -1 (o un valor grande para indicar "nunca visto").
    """
    logger.info("Calculando intervalo desde último aparecimiento...")
    columnas_numericas = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    df_intervalos = pd.DataFrame(index=df.index) # DataFrame para almacenar intervalos

    last_seen_index = {numero: -1 for numero in range(0, 46)} # Inicializar índice del último sorteo visto para cada número

    for index, row in df.iterrows():
        intervalos_fila = {}
        for numero in range(0, 46):
            intervalos_fila[f'intervalo_n{numero:02d}'] = index - last_seen_index[numero] # Intervalo = Sorteo actual - Último sorteo visto

        for col_n in columnas_numericas:
            numero_sorteo = row[col_n]
            last_seen_index[numero_sorteo] = index # Actualizar el último índice visto para este número

        df_intervalos = pd.concat([df_intervalos, pd.DataFrame([intervalos_fila], index=[index])]) # Añadir intervalos de esta fila

    logger.info("Intervalo desde último aparecimiento calculado.")
    return df_intervalos


def crear_variables_temporales(df):
    """
    Crea variables temporales a partir de la columna 'fecha':
    - 'dia_semana': Día de la semana (0=Lunes, 6=Domingo).
    - 'mes': Mes del año (1-12).
    - 'anio': Año.
    - 'dia_mes': Día del mes (1-31).
    - 'trimestre': Trimestre del año (1-4).

    Args:
        df (pandas.DataFrame): DataFrame de sorteos con columna 'fecha' en formato datetime.

    Returns:
        pandas.DataFrame: DataFrame con las nuevas columnas temporales añadidas.
    """
    logger.info("Creando variables temporales...")
    df_temporal = df.copy()

    df_temporal['dia_semana'] = df_temporal['fecha'].dt.dayofweek
    df_temporal['mes'] = df_temporal['fecha'].dt.month
    df_temporal['anio'] = df_temporal['fecha'].dt.year
    df_temporal['dia_mes'] = df_temporal['fecha'].dt.day
    df_temporal['trimestre'] = df_temporal['fecha'].dt.quarter

    logger.info("Variables temporales creadas.")
    return df_temporal.drop(columns=['fecha']) # Eliminar columna 'fecha' original después de extraer características


def escalar_normalizar_features(df, columnas_numericas):
    """
    Escala y normaliza las columnas numéricas especificadas utilizando StandardScaler.

    Args:
        df (pandas.DataFrame): DataFrame con las columnas numéricas a escalar.
        columnas_numericas (list): Lista de nombres de columnas numéricas a escalar.

    Returns:
        pandas.DataFrame: DataFrame con las columnas numéricas escaladas.
        sklearn.preprocessing.StandardScaler: Objeto StandardScaler fiteado en los datos de entrada,
                                              para usarlo para escalar datos nuevos (validación, prueba, predicción).
    """
    logger.info("Escalando y normalizando características numéricas...")
    df_escalado = df.copy()
    scaler = StandardScaler()

    df_escalado[columnas_numericas] = scaler.fit_transform(df_escalado[columnas_numericas])
    logger.info("Características numéricas escaladas y normalizadas.")
    return df_escalado, scaler


def codificar_variables_categoricas(df):
    """
    Codifica la variable categórica 'modalidad' usando One-Hot Encoding.
    """
    logger.info("Codificando variable categórica 'modalidad'...")
    df_codificado = df.copy()
    encoder_modalidad = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    modalidad_encoded = encoder_modalidad.fit_transform(df_codificado[['modalidad']])

    # *** LÍNEAS DE LOGGING DEBUG AÑADIDAS - VERIFICAR QUE ESTÉN CORRECTAS ***
    logger.debug(f"encoder_modalidad: {encoder_modalidad}") # Inspeccionar el objeto encoder
    logger.debug(f"encoder_modalidad.feature_names_in_: {encoder_modalidad.feature_names_in_}") # Verificar feature_names_in_
    logger.debug(f"Input to get_feature_names_out: [['modalidad']]: [['modalidad']]") # Confirmar input esperado

    modalidad_df = pd.DataFrame(modalidad_encoded, columns=encoder_modalidad.get_feature_names_out(None)) # Pasar None como input_features
    df_codificado = pd.concat([df_codificado.reset_index(drop=True), modalidad_df], axis=1).drop('modalidad', axis=1)

    logger.info("Variable categórica 'modalidad' codificada.")
    return df_codificado, encoder_modalidad


def transformar_a_secuencias(df, secuencia_longitud, columnas_features, columna_target):
    """
    Transforma los datos en secuencias de tiempo para modelos RNN/LSTM/GRU.

    Args:
        df (pandas.DataFrame): DataFrame preprocesado.
        secuencia_longitud (int): Longitud de las secuencias de entrada.
        columnas_features (list): Lista de nombres de columnas a usar como características de entrada.
        columna_target (list): Lista de nombres de columnas a usar como variable objetivo (target).

    Returns:
        tuple: (secuencias_entrada, secuencias_salida) - Tuplas de arrays NumPy con las secuencias.
               - secuencias_entrada: Array NumPy de 3 dimensiones (muestras, longitud_secuencia, num_features).
               - secuencias_salida:  Array NumPy de 2 dimensiones (muestras, num_targets).
    """
    logger.info(f"Transformando datos a secuencias con longitud {secuencia_longitud}...")

    data_features = df[columnas_features].values
    data_target = df[columna_target].values

    secuencias_entrada = []
    secuencias_salida = []

    for i in range(len(data_features) - secuencia_longitud):
        secuencia_input = data_features[i : i + secuencia_longitud] # Secuencia de 'secuencia_longitud' registros como entrada
        secuencia_output = data_target[i + secuencia_longitud]     # El registro SIGUIENTE a la secuencia como salida (predicción)

        secuencias_entrada.append(secuencia_input)
        secuencias_salida.append(secuencia_output)

    secuencias_entrada = np.array(secuencias_entrada) # Convertir a NumPy arrays para eficiencia con PyTorch
    secuencias_salida = np.array(secuencias_salida)

    logger.info(f"Datos transformados a {len(secuencias_entrada)} secuencias.")
    return secuencias_entrada, secuencias_salida


def preprocess_pipeline(df_sorteos, secuencia_longitud):
    """
    Pipeline principal de preprocesamiento de datos. Orquesta todos los pasos.

    Args:
        df_sorteos (pandas.DataFrame): DataFrame de sorteos crudos.
        secuencia_longitud (int): Longitud de las secuencias para modelos RNN.

    Returns:
        tuple: (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor,
                scaler_features, encoder_modalidad)
               - Conjuntos de datos divididos y en formato Tensor de PyTorch.
               - Scaler de características numéricas.
               - Encoder OneHot de modalidad.
    """
    logger.info("Iniciando pipeline de preprocesamiento...")

    # 1. Limpieza de datos
    df_limpio = limpiar_datos(df_sorteos)

    # 2. Ingeniería de Características
    df_frecuencias = calcular_frecuencia_numeros(df_limpio.copy()) # Calcular frecuencias ANTES de otras features temporales
    df_intervalos = calcular_intervalo_ultimo_aparecimiento(df_limpio.copy())
    df_temporal = crear_variables_temporales(df_limpio.copy())

    # 3. Unir Features Ingenierizadas al DataFrame Principal
    df_preprocesado = pd.concat([df_limpio.reset_index(drop=True),
                                 df_frecuencias.reset_index(drop=True),
                                 df_intervalos.reset_index(drop=True),
                                 df_temporal.reset_index(drop=True)], axis=1)

    # 4. Codificación de variable categórica 'modalidad'
    df_codificado, encoder_modalidad = codificar_variables_categoricas(df_preprocesado.copy())

    # 5. Selección de Columnas para Features y Target (ADAPTAR SEGÚN MODELO)
    columnas_features_num = ['dia_semana', 'mes', 'anio', 'dia_mes', 'trimestre'] # Variables temporales numéricas
    columnas_features_freq_intervalo = list(df_frecuencias.columns) + list(df_intervalos.columns) # Frecuencias e Intervalos
    columnas_features_modalidad_ohe = list(filter(lambda col: col.startswith('modalidad_'), df_codificado.columns)) # Modalidad OHE
    columnas_features = columnas_features_num + columnas_features_freq_intervalo + columnas_features_modalidad_ohe # Combinar

    columna_target = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6'] # Números ganadores como target

    # 6. Escalado de Características Numéricas (SOLO las features numéricas, NO one-hot encoded ni target)
    df_escalado, scaler_features = escalar_normalizar_features(df_codificado.copy(), columnas_features_num + columnas_features_freq_intervalo) # Escalar numéricas + freq + intervalo

    # 7. Transformación a Secuencias (Para modelos RNN/LSTM/GRU - Opcional para otros modelos. Si no usas secuencias, comenta este paso y el siguiente)
    secuencias_entrada, secuencias_salida = transformar_a_secuencias(df_escalado.copy(), secuencia_longitud, columnas_features, columna_target)

    # 8. División de Datos (Temporal - ADAPTAR SEGÚN ESTRATEGIA DE VALIDACIÓN TEMPORAL)
    # from sklearn.model_selection import train_test_split  # ELIMINA ESTA IMPORTACIÓN SI ESTÁ PRESENTE - NO USAR DIVISION ALEATORIA
    def split_data(X: np.array, y: np.array, test_size=0.2, validation_size=0.2) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba usando DIVISIÓN TEMPORAL (CRONOLÓGICA).

        Args:
            X (np.array): Array NumPy de características (secuencias).
            y (np.array): Array NumPy de targets (bolas ganadoras).
            test_size (float, optional): Proporción del conjunto de prueba. Defaults to 0.2.
            validation_size (float, optional): Proporción del conjunto de validación. Defaults to 0.2.

        Returns:
            tuple[np.array, np.array, np.array, np.array, np.array, np.array]: Tupla con conjuntos:
                (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        logger.info("Dividiendo datos en conjuntos de entrenamiento, validación y prueba (DIVISIÓN TEMPORAL)...")

        total_samples = len(X)
        test_samples = int(total_samples * test_size) # Cantidad de muestras para prueba
        validation_samples = int(total_samples * validation_size) # Cantidad de muestras para validación
        train_samples = total_samples - test_samples - validation_samples # Cantidad para entrenamiento

        # División TEMPORAL: Tomar los primeros 'train_samples' para entrenamiento, los siguientes 'validation_samples' para validación, y los últimos 'test_samples' para prueba.
        X_train, X_val, X_test = np.split(X, [train_samples, train_samples + validation_samples]) # División por índices
        y_train, y_val, y_test = np.split(y, [train_samples, train_samples + validation_samples])

        logger.info(f"Conjuntos divididos temporalmente.")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    # 9. Convertir a Tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test) # Target tensor puede ser float o long según la loss function


    logger.info("Pipeline de preprocesamiento completado.")
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, scaler_features, encoder_modalidad


if __name__ == '__main__':
    from quini6_pipeline import data_ingestion, logging_config
    logging_config.setup_logging()

    engine = data_ingestion.conectar_db()
    if engine is None:
        print("Error al conectar a la base de datos. No se puede ejecutar el preprocesamiento de prueba.")
    else:
        df_sorteos_raw = data_ingestion.cargar_sorteos_db(engine)
        if df_sorteos_raw is not None:
            secuencia_longitud_test = 20 # Ejemplo de longitud de secuencia para prueba
            (X_train, y_train, X_val, y_val, X_test, y_test,
             scaler_features, encoder_modalidad) = preprocess_pipeline(df_sorteos_raw, secuencia_longitud_test)

            print("\n--- Datos Preprocesados (Ejemplo) ---")
            print("X_train Tensor shape:", X_train.shape)
            print("y_train Tensor shape:", y_train.shape)
            print("X_val Tensor shape:", X_val.shape)
            print("y_val Tensor shape:", y_val.shape)
            print("X_test Tensor shape:", X_test.shape)
            print("y_test Tensor shape:", y_test.shape)
            print("\nPrimeras 3 secuencias de X_train (ejemplo):\n", X_train[:3])
            print("\nPrimeras 3 targets de y_train (ejemplo):\n", y_train[:3])
            print("\nScaler Features (ejemplo):\n", scaler_features)
            print("\nEncoder Modalidad (ejemplo):\n", encoder_modalidad.categories_)


        else:
            print("Error al cargar sorteos de la base de datos. Preprocesamiento de prueba abortado.")