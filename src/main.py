import sys
import os
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import load_data
from src.data.data_processor import process_data
import pandas as pd
from src.data.data_splitter import split_data
from src.model.trainer import train_model
from src.model.evaluator import evaluate_model
from src.model.saver import save_model


def main():
    
    # Cargar los datos
    data = load_data(file_path = "data/raw/heart.csv")
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(data,target_column='target')
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    # Entrenar el modelo
    model = train_model(X_train=X_train, y_train=y_train)
    # print(type(model))

    # Evaluar el modelo
    accuracy, precision, recall, f1, auc = evaluate_model(model, test_data=X_test, y_test=y_test)
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1: {f1}")
    # print(f"AUC: {auc}")

    # Guardar el modelo
    save_model(model, model_path="models/trained_model")
    
if __name__ == "__main__":
    main()