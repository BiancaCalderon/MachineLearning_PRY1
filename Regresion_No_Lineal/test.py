import torch
import numpy as np
from Regresion_No_Lineal.models import RegressionModel
from Regresion_No_Lineal.dataset import SinDataset

def test_model(model_path='model.pth', num_samples=1000):
    # Cargar el modelo
    model = RegressionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Establecer el modelo en modo de evaluación

    # Crear el conjunto de datos para la evaluación
    test_dataset = SinDataset(num_samples)
    x_test = test_dataset.x
    y_test = test_dataset.y

    # Realizar predicciones
    with torch.no_grad():
        predictions = model(torch.tensor(x_test, dtype=torch.float32))

    # Calcular el error medio cuadrático
    mse = torch.mean((predictions - torch.tensor(y_test, dtype=torch.float32)) ** 2)
    print(f'Error medio cuadrático en el conjunto de prueba: {mse.item()}')

if __name__ == "__main__":
    test_model()
