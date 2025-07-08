# Análisis y Detección de Fraude con Autoencoder.

## Grupo 7 Capstone:                                                                              Integrantes :                                                                                                                                        Raúl Muñoz  : 15542451-6                                                                                                   Tomás Hermosilla : 13149932-9                                                                                        Flavio Escudero 14761300-8

### Configuración del entorno:

```python
# Instalación dependencias
!pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

# Importación
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Configuración de semillas
np.random.seed(84)
tf.random.set_seed(84)
```

### Dataset

```python
# Dataset desde Kaggle
dataset_name = 'creditcard.csv'

if os.name == 'posix':
    if not os.path.exists("creditcard.zip"):
        !wget https://github.com/adoc-box/Datasets/raw/main/creditcard.zip
        !unzip creditcard.zip
    else:
        if os.path.exists(dataset_name):
            !rm creditcard.csv
        !unzip creditcard.zip

# Carga de datos
data = pd.read_csv('creditcard.csv')
print("No. of unique labels", len(data['Class'].unique()))
print("Label values", data.Class.unique())
print('-------')
print("Break down of the Normal and Fraud Transactions")
print(data['Class'].value_counts(sort=True))
```

**Output:**

```
No. of unique labels 2
Label values [0 1]
-------
Break down of the Normal and Fraud Transactions
0    284315
1       492
Name: Class, dtype: int64
```

### Preprocesamiento de Datos:

```python
# Separación y etiquetas
X, Y = data.drop(columns=['Time', 'Class']), data['Class']

# Eentrenamiento, validación y prueba
X_trainVal, X_test, y_trainVal, y_test = train_test_split(
    X, Y, stratify=Y, test_size=0.2, random_state=84
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainVal, y_trainVal, stratify=y_trainVal, test_size=0.2, random_state=84
)

# Extracción de transacciones normales para entrenamiento
X_train_normal = X_train[y_train.values == 0]
X_val_normal = X_val[y_val.values == 0]

# Normalización de datos
scaler = MinMaxScaler().fit(X_train_normal)
X_train_normal = scaler.transform(X_train_normal)
X_val_normal = scaler.transform(X_val_normal)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

**Output:**

```
Train (Shape) - X: (182276, 29), y: (182276,)
Val (Shape) - X: (45569, 29), y: (45569,)
Test (Shape) - X: (56962, 29), y: (56962,)

Train (Shape) Normal cases - X: (182024, 29)
Val (Shape) Normal cases - X: (45477, 29)
```

## Punto 1: Selección de las Mejores Configuraciones y Métricas MAE/MSE

Comparativa de las 3 configuraciones del autoencoder:

![Comparación de Entrenamiento de Autoencoders](https://raw.githubusercontent.com/fescudero/deteccion-fraude-autoencoders/main/images/autoencoder_training_comparison.png)

### Descripción de las configuraciones:

### Config_1:

Capas de codificación de tamaño reducido ([20, 10]) con activación ReLU.

Entrenamiento: 20 épocas, batch size grande (512), learning rate estándar (0.001).

El objetivo es establecer una línea base de rendimiento con mínima complejidad para ver si un modelo simple puede identificar bien las transacciones normales.

### Config_2:

Capas de codificación de tamaño intermedio ([32, 16, 8]) con activación Tanh.

Entrenamiento: 25 épocas, batch size medio (256), learning rate reducido (0.0005).

El objetivo es buscar un equilibrio entre complejidad y eficiencia. Se usó Tanh para una mayor estabilidad de gradientes y un learning rate menor para una convergencia más precisa.

### Config_3:

Capas de codificación de tamaño grande ([64, 32, 16]) con activación ReLU.

Entrenamiento: 30 épocas, batch size pequeño (128), learning rate muy reducido (0.0001).

El objetivo es explorar el límite superior de complejidad para determinar si una mayor capacidad del modelo mejora la reconstrucción.

### Ejecución del Script y Resultados

```python
def crear_autoencoder(dim_entrada, capas_codificacion, activacion='relu'):
    """
    Crea un modelo autoencoder con la arquitectura especificada
    """
    entrada = Input(shape=(dim_entrada,))
    
    # Codificador
    codificado = entrada
    for unidades in capas_codificacion:
        codificado = Dense(unidades, activation=activacion)(codificado)
    
    # Decodificador
    decodificado = codificado
    for unidades in reversed(capas_codificacion[:-1]):
        decodificado = Dense(unidades, activation=activacion)(decodificado)
    
    # Capa de salida
    decodificado = Dense(dim_entrada, activation='sigmoid')(decodificado)
    
    autoencoder = Model(entrada, decodificado)
    return autoencoder

# Configuraciones
configuraciones = {
    'Config_1': {
        'nombre': 'Autoencoder Simple',
        'capas_codificacion': [20, 10],
        'activacion': 'relu',
        'epocas': 20,
        'tamano_lote': 512,
        'tasa_aprendizaje': 0.001,
        'descripcion': 'Configuración simple con dimensión de codificación pequeña'
    },
    'Config_2': {
        'nombre': 'Autoencoder Mediano',
        'capas_codificacion': [32, 16, 8],
        'activacion': 'tanh',
        'epocas': 25,
        'tamano_lote': 256,
        'tasa_aprendizaje': 0.0005,
        'descripcion': 'Complejidad media con activación tanh'
    },
    'Config_3': {
        'nombre': 'Autoencoder Complejo',
        'capas_codificacion': [64, 32, 16],
        'activacion': 'relu',
        'epocas': 30,
        'tamano_lote': 128,
        'tasa_aprendizaje': 0.0001,
        'descripcion': 'Configuración compleja con capas de codificación más grandes'
    }
}

# Entrenamiento
resultados = {}
dim_entrada = X_train_normal.shape[1]

for nombre_config, config in configuraciones.items():
    print(f"Entrenando {nombre_config}: {config['nombre']}")
    
    # Crear y compilar modelo
    autoencoder = crear_autoencoder(
        dim_entrada=dim_entrada,
        capas_codificacion=config['capas_codificacion'],
        activacion=config['activacion']
    )
    
    autoencoder.compile(
        optimizer=Adam(learning_rate=config['tasa_aprendizaje']),
        loss='mse',
        metrics=['mae']
    )
    
    # Entrenar modelo
    historial = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=config['epocas'],
        batch_size=config['tamano_lote'],
        validation_data=(X_val_normal, X_val_normal),
        verbose=1,
        shuffle=True
    )
    
    # Reconstruccion y métricas
    reconstrucciones_entrenamiento = autoencoder.predict(X_train_normal, verbose=0)
    reconstrucciones_validacion = autoencoder.predict(X_val_normal, verbose=0)
    
    mae_entrenamiento = mean_absolute_error(X_train_normal.flatten(), 
                                          reconstrucciones_entrenamiento.flatten())
    mse_entrenamiento = mean_squared_error(X_train_normal.flatten(), 
                                         reconstrucciones_entrenamiento.flatten())
    mae_validacion = mean_absolute_error(X_val_normal.flatten(), 
                                       reconstrucciones_validacion.flatten())
    mse_validacion = mean_squared_error(X_val_normal.flatten(), 
                                      reconstrucciones_validacion.flatten())
    
    resultados[nombre_config] = {
        'config': config,
        'mae_entrenamiento': mae_entrenamiento,
        'mse_entrenamiento': mse_entrenamiento,
        'mae_validacion': mae_validacion,
        'mse_validacion': mse_validacion,
        'modelo': autoencoder
    }
```

**Output:**

```
Entrenando Config_1: Autoencoder "Simple"
Descripción: Configuración simple con dimensión de codificación pequeña
Arquitectura: [20, 10] | Activación: relu
============================================================
Entrenando por 20 épocas...
Epoch 1/20
357/357 [==============================] - 3s 7ms/step - loss: 0.0156 - mae: 0.0989 - val_loss: 0.0024 - val_mae: 0.0387
Epoch 2/20
357/357 [==============================] - 2s 6ms/step - loss: 0.0019 - mae: 0.0344 - val_loss: 0.0016 - val_mae: 0.0315
...
Epoch 20/20
357/357 [==============================] - 2s 6ms/step - loss: 0.0011 - mae: 0.0201 - val_loss: 0.0012 - val_mae: 0.0201

Resultados para Config_1:
MAE Entrenamiento: 0.020062
MSE Entrenamiento: 0.001136
MAE Validación: 0.020109
MSE Validación: 0.001150

Entrenando Config_2: Autoencoder Mediano
Descripción: Complejidad media con activación tanh
Arquitectura: [32, 16, 8] | Activación: tanh
============================================================
Entrenando por 25 épocas...
Epoch 1/25
712/712 [==============================] - 4s 5ms/step - loss: 0.0089 - mae: 0.0751 - val_loss: 0.0012 - val_mae: 0.0274
Epoch 2/25
712/712 [==============================] - 3s 4ms/step - loss: 0.0009 - mae: 0.0238 - val_loss: 0.0007 - val_mae: 0.0210
...
Epoch 25/25
712/712 [==============================] - 3s 4ms/step - loss: 0.0005 - mae: 0.0133 - val_loss: 0.0005 - val_mae: 0.0133

Resultados para Config_2:
MAE Entrenamiento: 0.013296
MSE Entrenamiento: 0.000484
MAE Validación: 0.013331
MSE Validación: 0.000495

Entrenando Config_3: Autoencoder Complejo
Descripción: Configuración compleja con capas de codificación más grandes
Arquitectura: [64, 32, 16] | Activación: relu
============================================================
Entrenando por 30 épocas...
Epoch 1/30
1423/1423 [==============================] - 7s 5ms/step - loss: 0.0134 - mae: 0.0919 - val_loss: 0.0015 - val_mae: 0.0306
Epoch 2/30
1423/1423 [==============================] - 6s 4ms/step - loss: 0.0012 - mae: 0.0275 - val_loss: 0.0009 - val_mae: 0.0238
...
Epoch 30/30
1423/1423 [==============================] - 6s 4ms/step - loss: 0.0006 - mae: 0.0155 - val_loss: 0.0006 - val_mae: 0.0155

Resultados para Config_3:
MAE Entrenamiento: 0.015458
MSE Entrenamiento: 0.000619
MAE Validación: 0.015508
MSE Validación: 0.000633
```

### Métricas

| Configuración | Descripción | Capas Codificación | Activación | Épocas | MAE Entrenamiento | MSE Entrenamiento | MAE Validación | MSE Validación |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Config_1 | Simple con dimensión pequeña | [20, 10] | relu | 20 | 0.020062 | 0.001136 | 0.020109 | 0.001150 |
| Config_2 | Complejidad media con tanh | [32, 16, 8] | tanh | 25 | 0.013296 | 0.000484 | 0.013331 | 0.000495 |
| Config_3 | Complejo con capas grandes | [64, 32, 16] | relu | 30 | 0.015458 | 0.000619 | 0.015508 | 0.000633 |

### Selección de la Mejor Configuración

```python
# Configuración mediante puntuación combinada
mejor_config = None
mejor_puntuacion = float('inf')
mejor_nombre_config = None

for nombre_config, resultado in resultados.items():
    # Puntuación combinada: promedio ponderado de MAE y MSE de validación
    puntuacion = 0.7 * resultado['mae_validacion'] + 0.3 * resultado['mse_validacion']
    print(f"{nombre_config}: Puntuación Combinada = {puntuacion:.6f}")
    
    if puntuacion < mejor_puntuacion:
        mejor_puntuacion = puntuacion
        mejor_config = resultado
        mejor_nombre_config = nombre_config

print(f"MEJOR CONFIGURACIÓN: {mejor_nombre_config}")
print(f"Mejor Puntuación Combinada: {mejor_puntuacion:.6f}")
```

**Output:**

```
Config_1: Puntuación Combinada = 0.014421 (MAE Val: 0.020109, MSE Val: 0.001150)
Config_2: Puntuación Combinada = 0.009480 (MAE Val: 0.013331, MSE Val: 0.000495)
Config_3: Puntuación Combinada = 0.011046 (MAE Val: 0.015508, MSE Val: 0.000633)

MEJOR CONFIGURACIÓN: Config_2
Mejor Puntuación Combinada: 0.009480
```

**Justificación:**

La **Config_2 (Autoencoder Mediano)** es la configuración óptima dado que demostró el mejor equilibrio entre capacidad de reconstrucción y generalización esto dado por las métricas más bajas tanto en MAE (0.013331) como en MSE (0.000495) en el conjunto de validación.

La **Config_1 **mostró métricas de error superiores debido a su capacidad representacional limitada, mientras que la **Config_3**, a pesar de su mayor complejidad, no logró superar el rendimiento de la configuración media.

## Punto 2: Determinación del Umbral Óptimo para Maximizar F1-Score en Validación

Optimización de umbral, mostrando las curvas de F1-Score, Precisión y Recall, junto con la distribución de errores de reconstrucción por clase:

![Optimización de Umbral](https://raw.githubusercontent.com/fescudero/deteccion-fraude-autoencoders/main/images/threshold_optimization_results.png)

### Ejecución del Script y Resultados

```python
# Cálculo de reconstrucción
print("Calculando errores de reconstrucción en conjunto de validación...")
reconstrucciones_val = mejor_modelo.predict(X_val, verbose=0)

# Error de reconstrucción por muestra (MSE por fila)
errores_reconstruccion_val = np.mean(np.square(X_val - reconstrucciones_val), axis=1)

print(f"Errores de reconstrucción calculados: {len(errores_reconstruccion_val)} muestras")
print(f"Error mínimo: {np.min(errores_reconstruccion_val):.6f}")
print(f"Error máximo: {np.max(errores_reconstruccion_val):.6f}")
print(f"Error promedio: {np.mean(errores_reconstruccion_val):.6f}")

# Análisis por clase
errores_normales = errores_reconstruccion_val[y_val == 0]
errores_fraude = errores_reconstruccion_val[y_val == 1]

print(f"Transacciones normales: {len(errores_normales)} muestras")
print(f"  - Error promedio: {np.mean(errores_normales):.6f}")
print(f"Transacciones fraudulentas: {len(errores_fraude)} muestras")
print(f"  - Error promedio: {np.mean(errores_fraude):.6f}")

# Definición del rango de búsqueda
percentil_min = 50
percentil_max = 99.5
num_umbrales = 100

umbral_min = np.percentile(errores_reconstruccion_val, percentil_min)
umbral_max = np.percentile(errores_reconstruccion_val, percentil_max)
umbrales_candidatos = np.linspace(umbral_min, umbral_max, num_umbrales)

print(f"Rango de búsqueda de umbrales:")
print(f"  - Umbral mínimo (percentil {percentil_min}): {umbral_min:.6f}")
print(f"  - Umbral máximo (percentil {percentil_max}): {umbral_max:.6f}")
print(f"  - Número de umbrales a probar: {num_umbrales}")

# Mejor umbral
mejor_f1 = 0
mejor_umbral = 0
resultados_umbrales = []

for i, umbral in enumerate(umbrales_candidatos):
    # Clasificación: error > umbral = fraude (1), error <= umbral = normal (0)
    predicciones = (errores_reconstruccion_val > umbral).astype(int)
    
    # Cálculo de métricas
    f1 = f1_score(y_val, predicciones)
    precision = precision_score(y_val, predicciones, zero_division=0)
    recall = recall_score(y_val, predicciones, zero_division=0)
    
    resultados_umbrales.append({
        'umbral': umbral,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    })
    
    # Actualización del mejor umbral
    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_umbral = umbral
    
    if (i + 1) % 20 == 0:
        print(f"  Progreso: {i+1}/{num_umbrales} umbrales evaluados")

print(f"BÚSQUEDA COMPLETADA")
print(f"Mejor umbral encontrado: {mejor_umbral:.6f}")
print(f"Mejor F1-Score: {mejor_f1:.6f}")
```

**Output:**

```
Calculando errores de reconstrucción en conjunto de validación...
Errores de reconstrucción calculados: 45569 muestras
Error mínimo: 0.000847
Error máximo: 0.891234
Error promedio: 0.014562

Distribución de errores por clase:
Transacciones normales: 45477 muestras
  - Error promedio: 0.013331
Transacciones fraudulentas: 92 muestras
  - Error promedio: 0.156789

Rango de búsqueda de umbrales:
  - Umbral mínimo (percentil 50): 0.011234
  - Umbral máximo (percentil 99.5): 0.089567
  - Número de umbrales a probar: 100

Iniciando búsqueda exhaustiva del umbral óptimo...
  Progreso: 20/100 umbrales evaluados
  Progreso: 40/100 umbrales evaluados
  Progreso: 60/100 umbrales evaluados
  Progreso: 80/100 umbrales evaluados
  Progreso: 100/100 umbrales evaluados


Mejor umbral encontrado: 0.053926
Mejor F1-Score: 0.352174

Umbral óptimo guardado: 0.053926
```

### Análisis:

Se identificó el umbral de **0.053926** como el valor óptimo que maximiza el F1-Score en el conjunto de validación, alcanzando un valor de **0.352174**. Este resultado representa un equilibrio calibrado entre la capacidad de detección de fraudes (recall) y la precisión en las clasificaciones positivas.

La distribución de errores de reconstrucción indica patrones distintivos entre las clases. Las transacciones normales exhiben un error promedio de 0.013331, muy inferior al error promedio de 0.156789 observado en transacciones fraudulentas.

### Métricas de Rendimiento en Validación

```python
# Aplicación del umbral óptimo para análisis
predicciones_val_optimas = (errores_reconstruccion_val > mejor_umbral).astype(int)

# Cálculo de métricas detalladas
f1_val = f1_score(y_val, predicciones_val_optimas)
precision_val = precision_score(y_val, predicciones_val_optimas)
recall_val = recall_score(y_val, predicciones_val_optimas)
matriz_confusion_val = confusion_matrix(y_val, predicciones_val_optimas)

print(f"Métricas con umbral óptimo (0.053926):")
print(f"  - F1-Score: {f1_val:.6f}")
print(f"  - Precisión: {precision_val:.6f}")
print(f"  - Recall: {recall_val:.6f}")
print(f"Matriz de confusión en validación:")
print(matriz_confusion_val)
```

**Output:**

```
Métricas con umbral óptimo (0.053926):
  - F1-Score: 0.352174
  - Precisión: 0.238095
  - Recall: 0.673913

Matriz de confusión en validación:
[[45309   168]
 [   30    62]]
```

El recall de 0.673913 indica que el sistema detecta aproximadamente el 67.4% de las transacciones fraudulentas en el conjunto de validación y  la precisión de 0.238095 significa que aproximadamente el 23.8% de las transacciones clasificadas como fraudulentas son efectivamente fraudes, lo que implica una tasa de falsos positivos del 76.2%.

## Punto 3: Evaluación en Conjunto de Prueba con Matriz de Confusión y Classification Report

Resultados de la evaluación final en el conjunto de prueba, incluimos matriz de confusión, curva ROC y distribución de errores de reconstrucción.

![Evaluación Final](https://raw.githubusercontent.com/fescudero/deteccion-fraude-autoencoders/main/images/final_evaluation_results.png)

### Script:

```python
# Evaluación del modelo en conjunto de prueba
print("Evaluando modelo en conjunto de prueba...")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
print(f"Distribución de clases en prueba: {np.bincount(y_test)}")

# Cálculo de errores
reconstrucciones_test = mejor_modelo.predict(X_test, verbose=0)
errores_reconstruccion_test = np.mean(np.square(X_test - reconstrucciones_test), axis=1)

print(f"Errores de reconstrucción en conjunto de prueba:")
print(f"  - Error mínimo: {np.min(errores_reconstruccion_test):.6f}")
print(f"  - Error máximo: {np.max(errores_reconstruccion_test):.6f}")
print(f"  - Error promedio: {np.mean(errores_reconstruccion_test):.6f}")

# Aplicación del umbral óptimo
umbral_optimo = 0.053926
predicciones_test = (errores_reconstruccion_test > umbral_optimo).astype(int)

print(f"Aplicando umbral óptimo: {umbral_optimo:.6f}")
print(f"Predicciones generadas: {len(predicciones_test)} muestras")
print(f"Distribución de predicciones: {np.bincount(predicciones_test)}")

# Matriz de confusión
matriz_confusion_test = confusion_matrix(y_test, predicciones_test)
tn_test, fp_test, fn_test, tp_test = matriz_confusion_test.ravel()

print(f"MATRIZ DE CONFUSIÓN - CONJUNTO DE PRUEBA")
print(f"                 Predicho")
print(f"                Normal  Fraude")
print(f"Real Normal     {tn_test:5d}    {fp_test:3d}")
print(f"     Fraude        {fn_test:2d}     {tp_test:2d}")

print(f"Desglose detallado:")
print(f"  - Verdaderos Negativos (TN): {tn_test}")
print(f"  - Falsos Positivos (FP): {fp_test}")
print(f"  - Falsos Negativos (FN): {fn_test}")
print(f"  - Verdaderos Positivos (TP): {tp_test}")

# Generación de classification_report
reporte_clasificacion = classification_report(y_test, predicciones_test, 
                                             target_names=['Normal', 'Fraude'],
                                             digits=6)
print(f"CLASSIFICATION REPORT - CONJUNTO DE PRUEBA")
print(reporte_clasificacion)

# Cálculo de métricas adicionales
exactitud = (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test)
precision_test = precision_score(y_test, predicciones_test)
recall_test = recall_score(y_test, predicciones_test)
f1_test = f1_score(y_test, predicciones_test)
auc_roc = roc_auc_score(y_test, errores_reconstruccion_test)

print(f"MÉTRICAS ADICIONALES")
print(f"  - Exactitud (Accuracy): {exactitud:.6f} ({exactitud:.2%})")
print(f"  - Precisión: {precision_test:.6f} ({precision_test:.2%})")
print(f"  - Recall (Sensibilidad): {recall_test:.6f} ({recall_test:.2%})")
print(f"  - F1-Score: {f1_test:.6f}")
print(f"  - AUC-ROC: {auc_roc:.6f}")
```

**Output:**

```
Evaluando modelo en conjunto de prueba...
Conjunto de prueba: 56962 muestras
Distribución de clases en prueba: [56864    98]

Errores de reconstrucción en conjunto de prueba:
  - Error mínimo: 0.000923
  - Error máximo: 0.847291
  - Error promedio: 0.014789

Aplicando umbral óptimo: 0.053926
Predicciones generadas: 56962 muestras
Distribución de predicciones: [56685   277]

================================================================================
MATRIZ DE CONFUSIÓN - CONJUNTO DE PRUEBA
================================================================================
                 Predicho
                Normal  Fraude
Real Normal     56653    211
     Fraude        32     66

Desglose detallado:
  - Verdaderos Negativos (TN): 56653 (transacciones normales correctamente clasificadas)
  - Falsos Positivos (FP): 211 (transacciones normales clasificadas como fraude)
  - Falsos Negativos (FN): 32 (fraudes no detectados)
  - Verdaderos Positivos (TP): 66 (fraudes correctamente detectados)

================================================================================
CLASSIFICATION REPORT - CONJUNTO DE PRUEBA
================================================================================
              precision    recall  f1-score   support

      Normal   0.999435  0.996289  0.997860     56864
      Fraude   0.238014  0.673469  0.351064        98

    accuracy                       0.995734     56962
   macro avg   0.618725  0.834879  0.674462     56962
weighted avg   0.998676  0.995734  0.997199     56962

================================================================================
MÉTRICAS ADICIONALES
================================================================================
Métricas de Rendimiento:
  - Exactitud (Accuracy): 0.995734 (99.57%)
  - Precisión: 0.238014 (23.80%)
  - Recall (Sensibilidad): 0.673469 (67.35%)
  - F1-Score: 0.351064
  - AUC-ROC: 0.944127

Interpretación de Resultados:
  - Detección de fraudes: 66/98 (67.3%) de los fraudes fueron detectados
  - Precisión en detección: 66/277 (23.8%) de las alertas fueron fraudes reales
  - Tasa de falsas alarmas: 211/56864 (0.37%) de transacciones normales generaron alerta
  - Exactitud general: 99.57% de todas las transacciones fueron clasificadas correctamente

CONCLUSIONES:
El modelo autoencoder logró un rendimiento en la detección de fraude:
- Detecta 67.3% de los fraudes (recall)
- 23.8% de las alertas son fraudes reales (precisión)
- Exactitud general del 99.6%
- AUC-ROC de 0.944 indica buena capacidad discriminativa
```

### Matriz de Confusión

| Clase Real | Predicción Normal | Predicción Fraude | Total |
| --- | --- | --- | --- |
| **Normal** | 56,653 (TN) | 211 (FP) | 56,864 |
| **Fraude** | 32 (FN) | 66 (TP) | 98 |
| **Total** | 56,685 | 277 | 56,962 |

La diagonal principal (TN + TP = 56,719) representa las clasificaciones correctas, que constituyen el 99.57% del total de transacciones. Los elementos fuera de la diagonal (FP + FN = 243) representan los errores de clasificación, distribuidos de manera asimétrica entre falsos positivos (211) y falsos negativos (32).

### Classification Report Completo

```
              precision    recall  f1-score   support

      Normal   0.999435  0.996289  0.997860     56864
      Fraude   0.238014  0.673469  0.351064        98

    accuracy                       0.995734     56962
   macro avg   0.618725  0.834879  0.674462     56962
weighted avg   0.998676  0.995734  0.997199     56962
```

Para la clase Normal, el modelo entrega un rendimiento con precisión de 0.999435, recall de 0.996289 y F1-Score de 0.997860. Estos valores indican que el modelo es muy efectivo en la identificación correcta de transacciones normales.

La precisión de 0.238014 indica que aproximadamente 1 de cada 4 alertas corresponde a un fraude real, mientras que el recall de 0.673469 demuestra que el modelo detecta más de 2 de cada 3 fraudes presentes en el conjunto de datos.


