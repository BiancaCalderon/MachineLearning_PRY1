import torch
from Clasificacion_Digitos_MNIST.models import DigitClassificationModel
from Clasificacion_Digitos_MNIST.dataset import get_dataloaders

def test_model():
    _, test_loader = get_dataloaders()
    model = DigitClassificationModel()
    model.load_state_dict(torch.load("Clasificacion_Digitos_MNIST/mnist_model.pth"))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.view(x.shape[0], -1)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Precisión en el conjunto de prueba: {accuracy:.2f}%")
