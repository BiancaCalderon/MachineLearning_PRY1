# Regresión No Lineal

## 📌 Descripción
En este módulo implementaremos un modelo de **Regresión No Lineal** utilizando **redes neuronales en PyTorch**. El objetivo es aproximar la función `sin(x)` en el intervalo `[-2π, 2π]`.

## 📂 Estructura de archivos
```
Regresion_No_Lineal/
│── models.py        # Definición del modelo de regresión
│── dataset.py       # Generación y carga de datos
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
### 1️⃣ **models.py** - Definición del modelo
Completa la clase `RegressionModel` con los siguientes métodos:
- `__init__(self)`: Define una red neuronal con capas densas adecuadas.
- `forward(self, x)`: Implementa la propagación hacia adelante.
- `get_loss(self, x, y)`: Devuelve la pérdida MSE entre la predicción y el valor real.

### 2️⃣ **dataset.py** - Generación de datos
- Implementa `generate_data()` para crear un dataset de pares `(x, sin(x))` con `torch.Tensor`.

### 3️⃣ **train.py** - Entrenamiento del modelo
- Carga los datos generados.
- Define el optimizador y el proceso de entrenamiento.
- Asegura que el modelo se entrene hasta alcanzar un error `<= 0.02`.
- Ejecuta:
```bash
python train.py
```

### 4️⃣ **test.py** - Evaluación
- Implementa pruebas para medir el error de la regresión en datos de prueba.
- Ejecuta:
```bash
python test.py
```

## ✅ Criterios de Evaluación
Para considerar la implementación exitosa:
- El modelo debe alcanzar una **pérdida menor o igual a 0.02**.
- `autograder.py -q q2` debe aprobar la evaluación.

💡 **Consejo:** Si el entrenamiento no converge, ajusta la arquitectura o los hiperparámetros. ¡Mucho éxito! 🚀

