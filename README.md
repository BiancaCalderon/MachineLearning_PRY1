# MachineLearning_PRY1# Proyecto de Machine Learning - Inteligencia Artificial 2025

## 📌 Descripción General
Este proyecto implementa varios **modelos de aprendizaje automático** utilizando **PyTorch**, siguiendo los lineamientos del curso de **Inteligencia Artificial de la Universidad de Berkeley**. Se han dividido en **tres módulos**, cada uno con un objetivo específico:

1. **Perceptrón Binario**: Implementa un clasificador lineal binario.
2. **Regresión No Lineal**: Entrena una red neuronal para aproximar la función `sin(x)`.
3. **Clasificación de Dígitos MNIST**: Usa una red neuronal para clasificar imágenes de dígitos escritos a mano.

## 📂 Estructura del Proyecto
```
Proyecto_ML/
│── Perceptron_Binario/         # Implementación del Perceptrón
│── Regresion_No_Lineal/        # Implementación de la regresión no lineal
│── Clasificacion_Digitos_MNIST/ # Clasificación de dígitos MNIST
│── requirements.txt            # Dependencias del proyecto
│── autograder.py               # Script de evaluación automática
│── main.py                     # Punto de entrada general
│── .gitignore                   # Archivos a ignorar en Git
│── README.md                    # Documentación global
```

## 🛠️ Instalación de dependencias
Asegúrate de tener **Python 3.8+** instalado y ejecuta:
```bash
pip install -r requirements.txt
```

## 🚀 Cómo ejecutar cada módulo
Cada carpeta tiene su propio `README.md` con instrucciones detalladas. Sin embargo, en general, puedes ejecutar:

### 🔹 **Entrenar un modelo**
```bash
cd NombreDelModulo
python train.py
```
Ejemplo:
```bash
cd Perceptron_Binario
python train.py
```

### 🔹 **Probar un modelo**
```bash
cd NombreDelModulo
python test.py
```
Ejemplo:
```bash
cd Clasificacion_Digitos_MNIST
python test.py
```

### 🔹 **Ejecutar el evaluador automático**
Para validar las implementaciones, usa:
```bash
python autograder.py -q qX
```
Donde `qX` es el número de la pregunta (1 para Perceptrón, 2 para Regresión, 3 para MNIST).

## ✅ Criterios de Evaluación
- **Perceptrón**: Debe alcanzar **100% precisión** en el conjunto de entrenamiento.
- **Regresión No Lineal**: Debe lograr una **pérdida ≤ 0.02**.
- **Clasificación MNIST**: Debe obtener **≥ 97% precisión en el conjunto de prueba**.

## 🏗️ Contribuciones y Colaboración
- Usa **branching en Git** para cada módulo.
- Cada cambio debe ser revisado mediante **pull requests**.
- Sigue el formato de codificación del curso.

💡 **Consejo:** Revisa los `README.md` individuales para más detalles sobre cada módulo. ¡Mucho éxito en la implementación! 🚀

