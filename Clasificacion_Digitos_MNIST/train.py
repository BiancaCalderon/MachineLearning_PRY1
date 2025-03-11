import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from Clasificacion_Digitos_MNIST.models import DigitClassificationModel
from Clasificacion_Digitos_MNIST.dataset import get_dataloaders

def train_model(epochs=5, learning_rate=0.001, batch_size=64):
    train_loader, _ = get_dataloaders(batch_size)
    model = DigitClassificationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Cambio a AdamW

    for epoch in range(epochs):
        correct, total = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}")
        for batch in progress_bar:
            x, y = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        accuracy = 100 * correct / total
        print(f"Época {epoch+1}/{epochs}, Precisión: {accuracy:.2f}%")

    torch.save(model.state_dict(), "Clasificacion_Digitos_MNIST/mnist_model.pth")
    print("Modelo guardado exitosamente.")
