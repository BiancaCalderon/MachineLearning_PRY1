import torch
from torchvision import datasets, transforms

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.RandomRotation(15),  # Más variabilidad
        transforms.RandomAffine(0, shear=15, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Valores optimizados para MNIST
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
