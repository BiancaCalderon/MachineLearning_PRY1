# Perceptron Binario

## 📌 Descripción
Este módulo implementa un **Perceptrón Binario** utilizando **PyTorch**. Un perceptrón es un modelo de clasificación supervisado que ajusta pesos en función de un conjunto de entrenamiento para clasificar datos en dos categorías (-1 o 1).

## 📂 Estructura de archivos
```
Perceptron_Binario/
│── models.py        # Definición del modelo Perceptrón
│── dataset.py       # Carga y preprocesamiento de datos
│── train.py         # Entrenamiento del modelo
│── test.py          # Evaluación del modelo
│── README.md        # Esta documentación
```

## 🛠️ Instalación de dependencias
Asegúrate de tener **Python 3.8+** y ejecuta:
```bash
pip install -r requirements.txt
```

## 🚀 Instrucciones de Implementación
### 1️⃣ **models.py** - Implementación del modelo
Completa la clase `PerceptronModel` con los siguientes métodos:
- `__init__(self, dimensions)`: Inicializa los pesos del perceptrón como un tensor `Parameter` de PyTorch.
- `run(self, x)`: Calcula el producto escalar entre los pesos y la entrada.
- `get_prediction(self, x)`: Retorna `1` si el producto escalar es positivo, `-1` si es negativo.
- `train(self, dataset, epochs)`: Ajusta los pesos iterando sobre el dataset hasta lograr un 100% de precisión.

### 2️⃣ **dataset.py** - Carga de datos
- Implementa la función `load_data()` para cargar los datos en un **DataLoader** de PyTorch.

### 3️⃣ **train.py** - Entrenamiento
- Importa `PerceptronModel` y `load_data()`.
- Define el flujo de entrenamiento, asegurándote de actualizar los pesos correctamente.
- Ejecuta:
```bash
python train.py
```

### 4️⃣ **test.py** - Evaluación
- Implementa pruebas para validar la precisión del modelo.
- Ejecuta:
```bash
python test.py
```

## ✅ Criterios de Evaluación
Para considerar la implementación exitosa:
- El modelo debe alcanzar **100% de precisión** en el conjunto de entrenamiento.
- `autograder.py -q q1` debe aprobar la evaluación.

💡 **Sugerencia:** Usa `print()` y `assert` para depuración. ¡Buena suerte! 🚀

