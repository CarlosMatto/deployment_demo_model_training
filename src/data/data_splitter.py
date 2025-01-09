from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data: pd.DataFrame, target_column: str, test_size=0.25, 
               random_state=42, stratify: bool = True
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos escalados en conjuntos de entrenamiento y prueba.

    Args:
        scaled_data (pd.DataFrame): Datos ya procesados y escalados.
        target_column (str): El nombre de la columna objetivo.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla para la reproducibilidad.
        stratify (bool): Si se debe estratificar por la variable objetivo.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separar características y objetivo
    X = data.drop(columns=target_column, axis=1)  # Usar scaled_data aquí
    y = data[target_column]

    # Dividir en conjuntos de entrenamiento y prueba
    if stratify:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    
    return x_train, x_test, y_train, y_test