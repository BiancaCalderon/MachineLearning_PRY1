import torch
import torch.nn as nn
import torch.optim as optim

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Definir la arquitectura de la red
        self.fc1 = nn.Linear(1, 10)  # Capa oculta
        self.fc2 = nn.Linear(10, 1)   # Capa de salida

    def forward(self, x):
        # Propagación hacia adelante
        x = torch.relu(self.fc1(x))   # Aplicar ReLU
        x = self.fc2(x)                # Capa de salida
        return x

    def get_loss(self, x, y):
        # Calcular la pérdida
        predictions = self.forward(x)
        loss = nn.MSELoss()(predictions, y)  # Pérdida MSE
        return loss

    def train_model(self, train_loader, num_epochs=1000, target_loss=0.02):
        optimizer = optim.Adam(self.parameters(), lr=0.01)  # Optimizador
        for epoch in range(num_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()  # Gradientes a cero
                loss = self.get_loss(x, y)  # Calcular pérdida
                loss.backward()  # Retropropagación
                optimizer.step()  # Actualizar pesos

                if loss.item() < target_loss:
                    print(f"Entrenamiento detenido en la época {epoch} con pérdida {loss.item()}")
                    return
