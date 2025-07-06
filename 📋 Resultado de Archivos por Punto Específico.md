# 📋 Mapeo de Archivos por Punto Específico

## Punto 1: Selección de 3 mejores configuraciones y métricas MAE/MSE

### 🔧 Implementación:

- **`autoencoder_simple.py`** - Código principal que implementa las 3 configuraciones

### 📊 Resultados:

- **`autoencoder_comparison_table.csv`** - Tabla comparativa con métricas MAE/MSE

- **`best_autoencoder_config.json`** - Configuración seleccionada como mejor

- **`autoencoder_training_comparison.png`** - Visualización comparativa

---

## Punto 2: Encontrar mejor umbral para maximizar F1-Score

### 🔧 Implementación:

- **`threshold_optimization.py`** - Código de búsqueda de umbral óptimo

### 📊 Resultados:

- **`optimal_threshold_info.json`** - Umbral óptimo y métricas

- **`threshold_search_results.csv`** - Resultados completos de búsqueda

- **`threshold_optimization_results.png`** - Visualización de optimización

---

## Punto 3: Evaluación en conjunto de prueba

### 🔧 Implementación:

- **`final_evaluation.py`** - Código de evaluación final en conjunto de prueba

### 📊 Resultados:

- **`final_evaluation_results.json`** - Matriz de confusión y métricas finales

- **`final_evaluation_results.png`** - Visualización de resultados finales

---

## 📁 Archivos de Configuración:

- **`README.md`** 

- **`requirements.txt`** - Dependencias

- **`LICENSE`** 

---

## 🎯 Resumen de Resultados:

### Punto 1 - Mejor Configuración:

- **Config_2 (Autoencoder Mediano)** seleccionado

- Capas: [32, 16, 8] con activación tanh

- MAE Validación: 0.013331, MSE Validación: 0.000495

### Punto 2 - Umbral Óptimo:

- **Umbral: 0.053926**

- **F1-Score máximo: 0.352** en validación

### Punto 3 - Evaluación Final:

- **F1-Score: 0.352** en conjunto de prueba

- **Exactitud: 99.57%**

- **AUC-ROC: 0.944**

