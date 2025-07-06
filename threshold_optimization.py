# Optimización de Umbral para Detección de Fraude
# Fase 3: Determinación de umbral óptimo
# Autor: Grupo 7 Capstone MIA 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import json

# Configurar semillas aleatorias para reproducibilidad
np.random.seed(84)
tf.random.set_seed(84)

print("Iniciando optimización de umbral para detección de fraude...")
print(f"Versión de TensorFlow: {tf.__version__}")

# Cargar datos preprocesados
print("Cargando datos preprocesados...")
X_val_scaled = np.load('X_val_scaled.npy')
y_val = np.load('y_val.npy')

print(f"Forma de datos de validación: {X_val_scaled.shape}")
print(f"Forma de etiquetas de validación: {y_val.shape}")
print(f"Distribución de clases en conjunto de validación:")
print(f"Transacciones normales: {np.sum(y_val == 0)}")
print(f"Transacciones fraudulentas: {np.sum(y_val == 1)}")

# Cargar configuración del mejor modelo
with open('best_autoencoder_config.json', 'r', encoding='utf-8') as f:
    info_mejor_config = json.load(f)

print(f"\nMejor configuración de modelo: {info_mejor_config['mejor_nombre_config']}")
print(f"Mejor puntuación combinada: {info_mejor_config['mejor_puntuacion']:.6f}")

# Recrear el mejor modelo autoencoder
def crear_autoencoder(dim_entrada, capas_codificacion, activacion='relu'):
    # Capa de entrada
    capa_entrada = keras.Input(shape=(dim_entrada,))
    
    # Codificador
    codificado = capa_entrada
    for unidades in capas_codificacion:
        codificado = keras.layers.Dense(unidades, activation=activacion)(codificado)
    
    # Decodificador (espejo del codificador)
    decodificado = codificado
    for unidades in reversed(capas_codificacion[:-1]):
        decodificado = keras.layers.Dense(unidades, activation=activacion)(decodificado)
    
    # Capa de salida
    decodificado = keras.layers.Dense(dim_entrada, activation='sigmoid')(decodificado)
    
    # Crear modelo autoencoder
    autoencoder = keras.Model(capa_entrada, decodificado)
    
    return autoencoder

# Recrear y cargar el mejor modelo
dim_entrada = X_val_scaled.shape[1]
mejor_config = info_mejor_config['config']

print(f"\nRecreando mejor modelo autoencoder...")
print(f"Capas de codificación: {mejor_config['capas_codificacion']}")
print(f"Activación: {mejor_config['activacion']}")

mejor_autoencoder = crear_autoencoder(
    dim_entrada=dim_entrada,
    capas_codificacion=mejor_config['capas_codificacion'],
    activacion=mejor_config['activacion']
)

# Compilar modelo (necesario para cargar pesos)
mejor_autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=mejor_config['tasa_aprendizaje']),
    loss='mse',
    metrics=['mae']
)

# Cargar los pesos entrenados
archivo_pesos = f"autoencoder_{info_mejor_config['mejor_nombre_config'].lower()}.weights.h5"
print(f"Cargando pesos desde: {archivo_pesos}")
mejor_autoencoder.load_weights(archivo_pesos)

# Calcular errores de reconstrucción en conjunto de validación
print("\nCalculando errores de reconstrucción en conjunto de validación...")
reconstrucciones_val = mejor_autoencoder.predict(X_val_scaled, verbose=0)

# Calcular errores de reconstrucción (usando MAE ya que es más interpretable)
errores_reconstruccion = np.mean(np.abs(X_val_scaled - reconstrucciones_val), axis=1)

print(f"Estadísticas de errores de reconstrucción:")
print(f"Media: {np.mean(errores_reconstruccion):.6f}")
print(f"Desviación estándar: {np.std(errores_reconstruccion):.6f}")
print(f"Mínimo: {np.min(errores_reconstruccion):.6f}")
print(f"Máximo: {np.max(errores_reconstruccion):.6f}")

# Separar errores por clase para análisis
errores_normales = errores_reconstruccion[y_val == 0]
errores_fraude = errores_reconstruccion[y_val == 1]

print(f"\nErrores de reconstrucción por clase:")
print(f"Transacciones normales - Media: {np.mean(errores_normales):.6f}, Desv. Est.: {np.std(errores_normales):.6f}")
print(f"Transacciones fraudulentas - Media: {np.mean(errores_fraude):.6f}, Desv. Est.: {np.std(errores_fraude):.6f}")

# Búsqueda de umbral
print(f"\n{'='*60}")
print("OPTIMIZACIÓN DE UMBRAL")
print(f"{'='*60}")

# Definir rango de umbral basado en distribución de errores de reconstrucción
umbral_min = np.percentile(errores_reconstruccion, 50)  # Comenzar desde la mediana
umbral_max = np.percentile(errores_reconstruccion, 99.5)  # Hasta el percentil 99.5
rango_umbrales = np.linspace(umbral_min, umbral_max, 100)

print(f"Buscando umbrales desde {umbral_min:.6f} hasta {umbral_max:.6f}")

# Almacenar resultados para cada umbral
resultados_umbral = []

mejor_f1 = 0
mejor_umbral = 0
mejores_metricas = {}

for umbral in rango_umbrales:
    # Clasificar: fraude si error de reconstrucción > umbral
    y_pred = (errores_reconstruccion > umbral).astype(int)
    
    # Calcular métricas
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    
    resultados_umbral.append({
        'umbral': umbral,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    })
    
    # Rastrear mejor puntuación F1
    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_umbral = umbral
        mejores_metricas = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'y_pred': y_pred
        }

print(f"\nUmbral óptimo encontrado: {mejor_umbral:.6f}")
print(f"Mejor F1-Score: {mejor_f1:.6f}")
print(f"Precisión: {mejores_metricas['precision']:.6f}")
print(f"Recall: {mejores_metricas['recall']:.6f}")

# Generar reporte de clasificación detallado para el mejor umbral
print(f"\n{'='*60}")
print("REPORTE DE CLASIFICACIÓN PARA UMBRAL ÓPTIMO")
print(f"{'='*60}")

y_pred_mejor = mejores_metricas['y_pred']
print(classification_report(y_val, y_pred_mejor, target_names=['Normal', 'Fraude']))

# Matriz de confusión
mc = confusion_matrix(y_val, y_pred_mejor)
print(f"\nMatriz de Confusión:")
print(f"                 Predicho")
print(f"                Normal  Fraude")
print(f"Real Normal     {mc[0,0]:6d}  {mc[0,1]:5d}")
print(f"     Fraude     {mc[1,0]:6d}  {mc[1,1]:5d}")

# Guardar información de umbral óptimo
info_umbral_optimo = {
    'umbral_optimo': mejor_umbral,
    'mejor_f1_score': mejor_f1,
    'precision': mejores_metricas['precision'],
    'recall': mejores_metricas['recall'],
    'matriz_confusion': mc.tolist(),
    'rango_busqueda_umbral': {
        'umbral_min': umbral_min,
        'umbral_max': umbral_max,
        'num_umbrales_probados': len(rango_umbrales)
    },
    'estadisticas_error_reconstruccion': {
        'media_general': np.mean(errores_reconstruccion),
        'desv_est_general': np.std(errores_reconstruccion),
        'media_normal': np.mean(errores_normales),
        'desv_est_normal': np.std(errores_normales),
        'media_fraude': np.mean(errores_fraude),
        'desv_est_fraude': np.std(errores_fraude)
    }
}

with open('optimal_threshold_info.json', 'w', encoding='utf-8') as f:
    json.dump(info_umbral_optimo, f, indent=2, ensure_ascii=False)

print(f"\nInformación de umbral óptimo guardada como 'optimal_threshold_info.json'")

# Crear visualizaciones
plt.figure(figsize=(15, 12))

# Gráfico 1: Distribución de errores de reconstrucción por clase
plt.subplot(2, 3, 1)
plt.hist(errores_normales, bins=50, alpha=0.7, label='Normal', density=True, color='blue')
plt.hist(errores_fraude, bins=50, alpha=0.7, label='Fraude', density=True, color='red')
plt.axvline(mejor_umbral, color='green', linestyle='--', linewidth=2, label=f'Umbral Óptimo: {mejor_umbral:.4f}')
plt.xlabel('Error de Reconstrucción (MAE)')
plt.ylabel('Densidad')
plt.title('Distribución de Errores de Reconstrucción por Clase')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: F1-Score vs Umbral
plt.subplot(2, 3, 2)
umbrales = [r['umbral'] for r in resultados_umbral]
f1_scores = [r['f1_score'] for r in resultados_umbral]
plt.plot(umbrales, f1_scores, 'b-', alpha=0.8)
plt.axvline(mejor_umbral, color='green', linestyle='--', linewidth=2, label=f'Óptimo: {mejor_umbral:.4f}')
plt.axhline(mejor_f1, color='red', linestyle='--', alpha=0.7, label=f'Mejor F1: {mejor_f1:.4f}')
plt.xlabel('Umbral')
plt.ylabel('F1-Score')
plt.title('F1-Score vs Umbral')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 3: Precisión vs Umbral
plt.subplot(2, 3, 3)
precisiones = [r['precision'] for r in resultados_umbral]
plt.plot(umbrales, precisiones, 'g-', alpha=0.8, label='Precisión')
plt.axvline(mejor_umbral, color='green', linestyle='--', linewidth=2)
plt.xlabel('Umbral')
plt.ylabel('Precisión')
plt.title('Precisión vs Umbral')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 4: Recall vs Umbral
plt.subplot(2, 3, 4)
recalls = [r['recall'] for r in resultados_umbral]
plt.plot(umbrales, recalls, 'r-', alpha=0.8, label='Recall')
plt.axvline(mejor_umbral, color='green', linestyle='--', linewidth=2)
plt.xlabel('Umbral')
plt.ylabel('Recall')
plt.title('Recall vs Umbral')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 5: Curva Precisión-Recall
plt.subplot(2, 3, 5)
plt.plot(recalls, precisiones, 'purple', alpha=0.8, linewidth=2)
plt.scatter(mejores_metricas['recall'], mejores_metricas['precision'], 
           color='green', s=100, zorder=5, label=f'Punto Óptimo')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.title('Curva Precisión-Recall')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 6: Mapa de calor de Matriz de Confusión
plt.subplot(2, 3, 6)
import seaborn as sns
sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraude'], 
            yticklabels=['Normal', 'Fraude'])
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')

plt.tight_layout()
plt.savefig('threshold_optimization_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualizaciones de optimización de umbral guardadas como 'threshold_optimization_results.png'")

# Guardar resultados de umbral para siguiente fase
df_resultados_umbral = pd.DataFrame(resultados_umbral)
df_resultados_umbral.to_csv('threshold_search_results.csv', index=False)

print(f"Resultados de búsqueda de umbral guardados como 'threshold_search_results.csv'")
print(f"\n¡Fase 3 completada exitosamente!")
print(f"Umbral óptimo: {mejor_umbral:.6f} con F1-Score: {mejor_f1:.6f}")

# Guardar errores de reconstrucción para fase de prueba
np.save('val_reconstruction_errors.npy', errores_reconstruccion)
print("Errores de reconstrucción de validación guardados para siguiente fase")

