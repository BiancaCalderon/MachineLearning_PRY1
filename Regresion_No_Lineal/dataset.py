import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SinDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.x = np.linspace(-2 * np.pi, 2 * np.pi, num_samples).reshape(-1, 1)  # Entradas
        self.y = np.sin(self.x)  # Salidas

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def create_dataloader(batch_size=32, num_samples=1000):
    dataset = SinDataset(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
