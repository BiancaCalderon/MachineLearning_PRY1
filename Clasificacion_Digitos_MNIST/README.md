# Clasificación de Dígitos MNIST

## 📌 Descripción
Este módulo implementa una **red neuronal** en **PyTorch** para clasificar imágenes del conjunto de datos **MNIST** (dígitos escritos a mano). Se espera alcanzar al menos un **97% de precisión**.

## 📂 Estructura de archivos
```
Clasificacion_Digitos_MNIST/
│── models.py        # Definición del modelo neuronal
│── dataset.py       # Carga y preprocesamiento de datos MNIST
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
Completa la clase `DigitClassificationModel` con:
- `__init__(self)`: Define una red neuronal con capas densas o convolucionales.
- `forward(self, x)`: Propagación hacia adelante, retornando un vector de tamaño `batch_size × 10`.

### 2️⃣ **dataset.py** - Carga de datos MNIST
- Usa `torchvision.datasets.MNIST` para descargar y normalizar las imágenes.
- Define `get_dataloaders(batch_size)` para cargar los datos en PyTorch DataLoader.

### 3️⃣ **train.py** - Entrenamiento del modelo
- Define la función de pérdida `cross_entropy`.
- Usa `Adam` o `SGD` como optimizador.
- Entrena el modelo hasta superar el **97% de precisión en validación**.
- Ejecuta:
```bash
python train.py
```

### 4️⃣ **test.py** - Evaluación
- Implementa pruebas para medir la precisión del modelo en datos de prueba.
- Ejecuta:
```bash
python test.py
```

## ✅ Criterios de Evaluación
Para considerar la implementación exitosa:
- El modelo debe alcanzar **≥ 97% precisión en prueba**.
- `autograder.py -q q3` debe aprobar la evaluación.

💡 **Consejo:** Si la precisión es baja, ajusta la arquitectura o el número de épocas. ¡Mucho éxito! 🚀

