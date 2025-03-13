import torch
from torch.utils.data import DataLoader
from Regresion_No_Lineal.models import RegressionModel
from Regresion_No_Lineal.dataset import create_dataloader

def train_model(num_epochs=1000, target_loss=0.02, batch_size=32, num_samples=1000):
    # Crear DataLoader
    train_loader = create_dataloader(batch_size, num_samples)
    
    # Inicializar el modelo
    model = RegressionModel()
    
    # Entrenar el modelo
    model.train_model(train_loader, num_epochs, target_loss)

if __name__ == "__main__":
    train_model()
