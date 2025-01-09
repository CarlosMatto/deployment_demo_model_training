import pandas as pd
from sklearn.preprocessing import StandardScaler

# Definir y ajustar el escalador
scaler = StandardScaler()

def process_data(data, target_column='target', fit=False):
    """
    Preprocesa los datos antes de pasarlos al modelo.
    
    - Verifica que las columnas sean consistentes.
    - Aplica escalamiento a los datos.
    - Mantiene la columna target sin cambios.
    
    Args:
        data (pd.DataFrame): Datos de entrada con las características.
        target_column (str): Nombre de la columna de destino (target).
        fit (bool): Indica si se debe ajustar el escalador a los datos (True) o usar un escalador preajustado (False).
    
    Returns:
        pd.DataFrame: Datos escalados con la columna target intacta.
    """
    # Verificar columnas esperadas (sin incluir el target)
    expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                        'exang', 'oldpeak', 'slope', 'ca', 'thal']  # Reemplaza con tus columnas
    for col in expected_columns:
        if col not in data.columns:
            raise ValueError(f"Missing expected column: {col}")
    
    # Separar las características y la columna target
    X = data[expected_columns]  # Características
    y = data[target_column]     # Target

    # Ajustar el escalador si es necesario
    if fit:
        scaler.fit(X)
    
    # Aplicar escalamiento solo a las características
    data_scaled = scaler.transform(X)

    # Volver a combinar las características escaladas con el target
    scaled_data = pd.DataFrame(data_scaled, columns=expected_columns)
    scaled_data[target_column] = y  # Añadir la columna target sin cambios

    return scaled_data
