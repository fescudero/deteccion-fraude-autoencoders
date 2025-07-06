# Implementación Simple de Autoencoder para Detección de Fraude
# Fase 2: Optimización de hiperparámetros y selección de modelos
# Autor: Grupo 7 Capstone MIA 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Configurar semillas aleatorias para reproducibilidad
np.random.seed(84)
tf.random.set_seed(84)

print("Iniciando optimización de hiperparámetros de Autoencoder...")
print(f"Versión de TensorFlow: {tf.__version__}")

# Cargar datos preprocesados
print("Cargando datos preprocesados...")
X_train_normal_scaled = np.load('X_train_normal_scaled.npy')
X_val_normal_scaled = np.load('X_val_normal_scaled.npy')

print(f"Forma de datos de entrenamiento: {X_train_normal_scaled.shape}")
print(f"Forma de datos de validación: {X_val_normal_scaled.shape}")

# Función de Autoencoder Simple
def crear_autoencoder(dim_entrada, capas_codificacion, activacion='relu'):
    # Capa de entrada
    capa_entrada = keras.Input(shape=(dim_entrada,))
    
    # Codificador
    codificado = capa_entrada
    for unidades in capas_codificacion:
        codificado = layers.Dense(unidades, activation=activacion)(codificado)
    
    # Decodificador (espejo del codificador)
    decodificado = codificado
    for unidades in reversed(capas_codificacion[:-1]):
        decodificado = layers.Dense(unidades, activation=activacion)(decodificado)
    
    # Capa de salida
    decodificado = layers.Dense(dim_entrada, activation='sigmoid')(decodificado)
    
    # Crear modelo autoencoder
    autoencoder = keras.Model(capa_entrada, decodificado)
    
    return autoencoder

# Definir 3 configuraciones
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

# Almacenamiento de resultados
resultados = {}
dim_entrada = X_train_normal_scaled.shape[1]

# Entrenar y evaluar cada configuración
for nombre_config, config in configuraciones.items():
    print(f"\n{'='*60}")
    print(f"Entrenando {nombre_config}: {config['nombre']}")
    print(f"Descripción: {config['descripcion']}")
    print(f"{'='*60}")
    
    # Crear modelo autoencoder
    autoencoder = crear_autoencoder(
        dim_entrada=dim_entrada,
        capas_codificacion=config['capas_codificacion'],
        activacion=config['activacion']
    )
    
    # Compilar modelo
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['tasa_aprendizaje']),
        loss='mse',
        metrics=['mae']
    )
    
    # Entrenar modelo
    print(f"Entrenando por {config['epocas']} épocas...")
    historial = autoencoder.fit(
        X_train_normal_scaled, X_train_normal_scaled,
        epochs=config['epocas'],
        batch_size=config['tamano_lote'],
        validation_data=(X_val_normal_scaled, X_val_normal_scaled),
        verbose=1,
        shuffle=True
    )
    
    # Generar reconstrucciones para evaluación
    print("Generando reconstrucciones para evaluación...")
    reconstrucciones_entrenamiento = autoencoder.predict(X_train_normal_scaled, verbose=0)
    reconstrucciones_validacion = autoencoder.predict(X_val_normal_scaled, verbose=0)
    
    # Calcular métricas
    mae_entrenamiento = mean_absolute_error(X_train_normal_scaled.flatten(), reconstrucciones_entrenamiento.flatten())
    mse_entrenamiento = mean_squared_error(X_train_normal_scaled.flatten(), reconstrucciones_entrenamiento.flatten())
    
    mae_validacion = mean_absolute_error(X_val_normal_scaled.flatten(), reconstrucciones_validacion.flatten())
    mse_validacion = mean_squared_error(X_val_normal_scaled.flatten(), reconstrucciones_validacion.flatten())
    
    # Almacenar resultados
    resultados[nombre_config] = {
        'config': config,
        'mae_entrenamiento': mae_entrenamiento,
        'mse_entrenamiento': mse_entrenamiento,
        'mae_validacion': mae_validacion,
        'mse_validacion': mse_validacion,
        'historial': historial.history,
        'modelo': autoencoder
    }
    
    print(f"\nResultados para {nombre_config}:")
    print(f"MAE Entrenamiento: {mae_entrenamiento:.6f}")
    print(f"MSE Entrenamiento: {mse_entrenamiento:.6f}")
    print(f"MAE Validación: {mae_validacion:.6f}")
    print(f"MSE Validación: {mse_validacion:.6f}")
    
    # Guardar modelo
    autoencoder.save_weights(f'autoencoder_{nombre_config.lower()}.weights.h5')
    print(f"Pesos del modelo guardados como autoencoder_{nombre_config.lower()}.weights.h5")

# Crear tabla de comparación
print(f"\n{'='*80}")
print("TABLA DE COMPARACIÓN - CONFIGURACIONES DE AUTOENCODER")
print(f"{'='*80}")

datos_comparacion = []
for nombre_config, resultado in resultados.items():
    datos_comparacion.append({
        'Configuración': nombre_config,
        'Descripción': resultado['config']['descripcion'],
        'Capas Codificación': str(resultado['config']['capas_codificacion']),
        'Activación': resultado['config']['activacion'],
        'Épocas': resultado['config']['epocas'],
        'Tamaño Lote': resultado['config']['tamano_lote'],
        'Tasa Aprendizaje': resultado['config']['tasa_aprendizaje'],
        'MAE Entrenamiento': f"{resultado['mae_entrenamiento']:.6f}",
        'MSE Entrenamiento': f"{resultado['mse_entrenamiento']:.6f}",
        'MAE Validación': f"{resultado['mae_validacion']:.6f}",
        'MSE Validación': f"{resultado['mse_validacion']:.6f}"
    })

df_comparacion = pd.DataFrame(datos_comparacion)
print(df_comparacion.to_string(index=False))

# Guardar tabla de comparación
df_comparacion.to_csv('autoencoder_comparison_table.csv', index=False)
print(f"\nTabla de comparación guardada como 'autoencoder_comparison_table.csv'")

# Determinar mejor configuración
print(f"\n{'='*80}")
print("SELECCIÓN DE MEJOR CONFIGURACIÓN")
print(f"{'='*80}")

mejor_config = None
mejor_puntuacion = float('inf')
mejor_nombre_config = None

for nombre_config, resultado in resultados.items():
    # Puntuación combinada: promedio ponderado de MAE y MSE de validación
    puntuacion = 0.7 * resultado['mae_validacion'] + 0.3 * resultado['mse_validacion']
    print(f"{nombre_config}: Puntuación Combinada = {puntuacion:.6f} (MAE Val: {resultado['mae_validacion']:.6f}, MSE Val: {resultado['mse_validacion']:.6f})")
    
    if puntuacion < mejor_puntuacion:
        mejor_puntuacion = puntuacion
        mejor_config = resultado
        mejor_nombre_config = nombre_config

print(f"\nMEJOR CONFIGURACIÓN: {mejor_nombre_config}")
print(f"Mejor Puntuación Combinada: {mejor_puntuacion:.6f}")
print(f"Detalles de Configuración:")
print(f"  - Capas de Codificación: {mejor_config['config']['capas_codificacion']}")
print(f"  - Activación: {mejor_config['config']['activacion']}")
print(f"  - Épocas: {mejor_config['config']['epocas']}")
print(f"  - Tamaño de Lote: {mejor_config['config']['tamano_lote']}")
print(f"  - Tasa de Aprendizaje: {mejor_config['config']['tasa_aprendizaje']}")
print(f"  - MAE Validación: {mejor_config['mae_validacion']:.6f}")
print(f"  - MSE Validación: {mejor_config['mse_validacion']:.6f}")

# Guardar información del mejor modelo
info_mejor_modelo = {
    'mejor_nombre_config': mejor_nombre_config,
    'mejor_puntuacion': mejor_puntuacion,
    'config': mejor_config['config'],
    'metricas': {
        'mae_entrenamiento': mejor_config['mae_entrenamiento'],
        'mse_entrenamiento': mejor_config['mse_entrenamiento'],
        'mae_validacion': mejor_config['mae_validacion'],
        'mse_validacion': mejor_config['mse_validacion']
    }
}

with open('best_autoencoder_config.json', 'w', encoding='utf-8') as f:
    json.dump(info_mejor_modelo, f, indent=2, ensure_ascii=False)

print(f"\nInformación del mejor modelo guardada como 'best_autoencoder_config.json'")

# Crear visualización
plt.figure(figsize=(15, 10))

# Gráficos de historial de entrenamiento
for i, (nombre_config, resultado) in enumerate(resultados.items(), 1):
    plt.subplot(2, 3, i)
    historial = resultado['historial']
    
    plt.plot(historial['loss'], label='Pérdida Entrenamiento', alpha=0.8)
    if 'val_loss' in historial:
        plt.plot(historial['val_loss'], label='Pérdida Validación', alpha=0.8)
    
    plt.title(f'{nombre_config} - Historial de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Gráficos de barras de comparación
plt.subplot(2, 3, 4)
nombres_config = list(resultados.keys())
maes_entrenamiento = [resultados[nombre]['mae_entrenamiento'] for nombre in nombres_config]
maes_validacion = [resultados[nombre]['mae_validacion'] for nombre in nombres_config]

x = np.arange(len(nombres_config))
ancho = 0.35

plt.bar(x - ancho/2, maes_entrenamiento, ancho, label='MAE Entrenamiento', alpha=0.8)
plt.bar(x + ancho/2, maes_validacion, ancho, label='MAE Validación', alpha=0.8)
plt.xlabel('Configuración')
plt.ylabel('MAE')
plt.title('Comparación MAE')
plt.xticks(x, nombres_config, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
mses_entrenamiento = [resultados[nombre]['mse_entrenamiento'] for nombre in nombres_config]
mses_validacion = [resultados[nombre]['mse_validacion'] for nombre in nombres_config]

plt.bar(x - ancho/2, mses_entrenamiento, ancho, label='MSE Entrenamiento', alpha=0.8)
plt.bar(x + ancho/2, mses_validacion, ancho, label='MSE Validación', alpha=0.8)
plt.xlabel('Configuración')
plt.ylabel('MSE')
plt.title('Comparación MSE')
plt.xticks(x, nombres_config, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Comparación de puntuación combinada
plt.subplot(2, 3, 6)
puntuaciones_combinadas = [0.7 * resultados[nombre]['mae_validacion'] + 0.3 * resultados[nombre]['mse_validacion'] for nombre in nombres_config]
plt.bar(nombres_config, puntuaciones_combinadas, alpha=0.8, color='green')
plt.xlabel('Configuración')
plt.ylabel('Puntuación Combinada')
plt.title('Comparación Puntuación Combinada (Menor es Mejor)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoencoder_training_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nGráficos de comparación de entrenamiento guardados como 'autoencoder_training_comparison.png'")
print(f"\n¡Fase 2 completada exitosamente!")
print(f"Mejor modelo: {mejor_nombre_config} con puntuación combinada: {mejor_puntuacion:.6f}")

# Guardar todos los resultados para la siguiente fase
resultados_para_guardar = {}
for nombre_config, resultado in resultados.items():
    resultados_para_guardar[nombre_config] = {
        'config': resultado['config'],
        'mae_entrenamiento': resultado['mae_entrenamiento'],
        'mse_entrenamiento': resultado['mse_entrenamiento'],
        'mae_validacion': resultado['mae_validacion'],
        'mse_validacion': resultado['mse_validacion'],
        'historial': resultado['historial']
    }

np.save('autoencoder_results.npy', resultados_para_guardar, allow_pickle=True)
print("Todos los resultados guardados para la siguiente fase")

