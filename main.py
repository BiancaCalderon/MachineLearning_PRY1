import torch
from Clasificacion_Digitos_MNIST.train import train_model
from Clasificacion_Digitos_MNIST.test import test_model
from Regresion_No_Lineal.train import train_model
from Regresion_No_Lineal.test import test_model

if __name__ == "__main__":
    train_model()
    test_model()
