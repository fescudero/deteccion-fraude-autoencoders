# Evaluación Final del Modelo en Conjunto de Datos de Prueba
# Fase 4: Evaluación final del modelo
# Autor: Grupo 7 Capstone MIA 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import json
import seaborn as sns

# Configurar semillas aleatorias para reproducibilidad
np.random.seed(84)
tf.random.set_seed(84)

print("Iniciando evaluación final del modelo en conjunto de datos de prueba...")
print(f"Versión de TensorFlow: {tf.__version__}")

# Cargar datos de prueba
print("Cargando datos de prueba...")
X_test_scaled = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

print(f"Forma de datos de prueba: {X_test_scaled.shape}")
print(f"Forma de etiquetas de prueba: {y_test.shape}")
print(f"Distribución de clases en conjunto de prueba:")
print(f"Transacciones normales: {np.sum(y_test == 0)}")
print(f"Transacciones fraudulentas: {np.sum(y_test == 1)}")

# Cargar configuración del mejor modelo y umbral óptimo
with open('best_autoencoder_config.json', 'r', encoding='utf-8') as f:
    info_mejor_config = json.load(f)

with open('optimal_threshold_info.json', 'r', encoding='utf-8') as f:
    info_umbral = json.load(f)

umbral_optimo = info_umbral['umbral_optimo']

print(f"\nMejor configuración de modelo: {info_mejor_config['mejor_nombre_config']}")
print(f"Umbral óptimo: {umbral_optimo:.6f}")
print(f"F1-Score de validación: {info_umbral['mejor_f1_score']:.6f}")

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
dim_entrada = X_test_scaled.shape[1]
mejor_config = info_mejor_config['config']

print(f"\nRecreando mejor modelo autoencoder...")
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

# Calcular errores de reconstrucción en conjunto de prueba
print(f"\n{'='*60}")
print("CALCULANDO ERRORES DE RECONSTRUCCIÓN EN CONJUNTO DE PRUEBA")
print(f"{'='*60}")

reconstrucciones_prueba = mejor_autoencoder.predict(X_test_scaled, verbose=0)
errores_reconstruccion_prueba = np.mean(np.abs(X_test_scaled - reconstrucciones_prueba), axis=1)

print(f"Estadísticas de errores de reconstrucción de prueba:")
print(f"Media: {np.mean(errores_reconstruccion_prueba):.6f}")
print(f"Desviación estándar: {np.std(errores_reconstruccion_prueba):.6f}")
print(f"Mínimo: {np.min(errores_reconstruccion_prueba):.6f}")
print(f"Máximo: {np.max(errores_reconstruccion_prueba):.6f}")

# Separar errores por clase para análisis
errores_normales_prueba = errores_reconstruccion_prueba[y_test == 0]
errores_fraude_prueba = errores_reconstruccion_prueba[y_test == 1]

print(f"\nErrores de reconstrucción de prueba por clase:")
print(f"Transacciones normales - Media: {np.mean(errores_normales_prueba):.6f}, Desv. Est.: {np.std(errores_normales_prueba):.6f}")
print(f"Transacciones fraudulentas - Media: {np.mean(errores_fraude_prueba):.6f}, Desv. Est.: {np.std(errores_fraude_prueba):.6f}")

# Aplicar umbral óptimo para clasificación
print(f"\n{'='*60}")
print("APLICANDO UMBRAL ÓPTIMO PARA CLASIFICACIÓN")
print(f"{'='*60}")

y_pred_prueba = (errores_reconstruccion_prueba > umbral_optimo).astype(int)

# Calcular métricas finales
f1_prueba = f1_score(y_test, y_pred_prueba)
precision_prueba = precision_score(y_test, y_pred_prueba, zero_division=0)
recall_prueba = recall_score(y_test, y_pred_prueba, zero_division=0)
auc_prueba = roc_auc_score(y_test, errores_reconstruccion_prueba)

print(f"Resultados Finales de Prueba:")
print(f"F1-Score: {f1_prueba:.6f}")
print(f"Precisión: {precision_prueba:.6f}")
print(f"Recall: {recall_prueba:.6f}")
print(f"AUC-ROC: {auc_prueba:.6f}")

# Generar matriz de confusión
mc_prueba = confusion_matrix(y_test, y_pred_prueba)

print(f"\n{'='*60}")
print("MATRIZ DE CONFUSIÓN")
print(f"{'='*60}")
print(f"                 Predicho")
print(f"                Normal  Fraude")
print(f"Real Normal     {mc_prueba[0,0]:6d}  {mc_prueba[0,1]:5d}")
print(f"     Fraude     {mc_prueba[1,0]:6d}  {mc_prueba[1,1]:5d}")

# Calcular métricas adicionales
vn, fp, fn, vp = mc_prueba.ravel()
especificidad = vn / (vn + fp)
sensibilidad = vp / (vp + fn)  # Igual que recall
exactitud = (vp + vn) / (vp + vn + fp + fn)

print(f"\nMétricas Adicionales:")
print(f"Verdaderos Positivos (VP): {vp}")
print(f"Verdaderos Negativos (VN): {vn}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Falsos Negativos (FN): {fn}")
print(f"Exactitud: {exactitud:.6f}")
print(f"Sensibilidad (Recall): {sensibilidad:.6f}")
print(f"Especificidad: {especificidad:.6f}")

# Generar reporte de clasificación detallado
print(f"\n{'='*60}")
print("REPORTE DE CLASIFICACIÓN DETALLADO")
print(f"{'='*60}")
print(classification_report(y_test, y_pred_prueba, target_names=['Normal', 'Fraude']))

# Guardar resultados de evaluación final
resultados_finales = {
    'metricas_prueba': {
        'f1_score': f1_prueba,
        'precision': precision_prueba,
        'recall': recall_prueba,
        'auc_roc': auc_prueba,
        'exactitud': exactitud,
        'sensibilidad': sensibilidad,
        'especificidad': especificidad
    },
    'matriz_confusion': {
        'verdaderos_positivos': int(vp),
        'verdaderos_negativos': int(vn),
        'falsos_positivos': int(fp),
        'falsos_negativos': int(fn),
        'matriz': mc_prueba.tolist()
    },
    'estadisticas_error_reconstruccion': {
        'media_general': np.mean(errores_reconstruccion_prueba),
        'desv_est_general': np.std(errores_reconstruccion_prueba),
        'media_normal': np.mean(errores_normales_prueba),
        'desv_est_normal': np.std(errores_normales_prueba),
        'media_fraude': np.mean(errores_fraude_prueba),
        'desv_est_fraude': np.std(errores_fraude_prueba)
    },
    'configuracion_modelo': info_mejor_config,
    'umbral_optimo': umbral_optimo
}

with open('final_evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(resultados_finales, f, indent=2, ensure_ascii=False)

print(f"\nResultados de evaluación final guardados como 'final_evaluation_results.json'")

# Crear visualizaciones comprehensivas
plt.figure(figsize=(20, 15))

# Gráfico 1: Distribución de errores de reconstrucción de prueba por clase
plt.subplot(3, 4, 1)
plt.hist(errores_normales_prueba, bins=50, alpha=0.7, label='Normal', density=True, color='blue')
plt.hist(errores_fraude_prueba, bins=50, alpha=0.7, label='Fraude', density=True, color='red')
plt.axvline(umbral_optimo, color='green', linestyle='--', linewidth=2, label=f'Umbral: {umbral_optimo:.4f}')
plt.xlabel('Error de Reconstrucción (MAE)')
plt.ylabel('Densidad')
plt.title('Conjunto de Prueba: Distribución de Errores de Reconstrucción')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Mapa de calor de Matriz de Confusión
plt.subplot(3, 4, 2)
sns.heatmap(mc_prueba, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraude'], 
            yticklabels=['Normal', 'Fraude'])
plt.title('Conjunto de Prueba: Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')

# Gráfico 3: Curva ROC
plt.subplot(3, 4, 3)
fpr, tpr, _ = roc_curve(y_test, errores_reconstruccion_prueba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_prueba:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Conjunto de Prueba: Curva ROC')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Gráfico 4: Comparación de métricas (Validación vs Prueba)
plt.subplot(3, 4, 4)
metricas_val = [info_umbral['mejor_f1_score'], info_umbral['precision'], info_umbral['recall']]
metricas_prueba = [f1_prueba, precision_prueba, recall_prueba]
nombres_metricas = ['F1-Score', 'Precisión', 'Recall']

x = np.arange(len(nombres_metricas))
ancho = 0.35

plt.bar(x - ancho/2, metricas_val, ancho, label='Validación', alpha=0.8)
plt.bar(x + ancho/2, metricas_prueba, ancho, label='Prueba', alpha=0.8)
plt.xlabel('Métricas')
plt.ylabel('Puntuación')
plt.title('Rendimiento Validación vs Prueba')
plt.xticks(x, nombres_metricas)
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 5: Comparación de distribución de errores (Normal vs Fraude)
plt.subplot(3, 4, 5)
plt.boxplot([errores_normales_prueba, errores_fraude_prueba], labels=['Normal', 'Fraude'])
plt.axhline(umbral_optimo, color='green', linestyle='--', linewidth=2, label=f'Umbral: {umbral_optimo:.4f}')
plt.ylabel('Error de Reconstrucción')
plt.title('Conjunto de Prueba: Distribución de Errores por Clase')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 6: Gráfico de dispersión de errores de reconstrucción
plt.subplot(3, 4, 6)
indices_normales = np.where(y_test == 0)[0]
indices_fraude = np.where(y_test == 1)[0]

plt.scatter(indices_normales[:1000], errores_reconstruccion_prueba[indices_normales[:1000]], 
           alpha=0.6, s=1, label='Normal', color='blue')
plt.scatter(indices_fraude, errores_reconstruccion_prueba[indices_fraude], 
           alpha=0.8, s=10, label='Fraude', color='red')
plt.axhline(umbral_optimo, color='green', linestyle='--', linewidth=2, label=f'Umbral: {umbral_optimo:.4f}')
plt.xlabel('Índice de Muestra')
plt.ylabel('Error de Reconstrucción')
plt.title('Conjunto de Prueba: Errores de Reconstrucción')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 7: Gráfico de barras de métricas de rendimiento
plt.subplot(3, 4, 7)
metricas = ['F1-Score', 'Precisión', 'Recall', 'Exactitud', 'Especificidad', 'AUC-ROC']
valores = [f1_prueba, precision_prueba, recall_prueba, exactitud, especificidad, auc_prueba]
colores = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange']

barras = plt.bar(metricas, valores, color=colores, alpha=0.8)
plt.ylabel('Puntuación')
plt.title('Conjunto de Prueba: Todas las Métricas de Rendimiento')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Agregar etiquetas de valores en las barras
for barra, valor in zip(barras, valores):
    plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.01, 
             f'{valor:.3f}', ha='center', va='bottom', fontsize=8)

# Gráfico 8: Matriz de confusión en porcentajes
plt.subplot(3, 4, 8)
mc_porcentaje = mc_prueba.astype('float') / mc_prueba.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(mc_porcentaje, annot=True, fmt='.1f', cmap='Blues', 
            xticklabels=['Normal', 'Fraude'], 
            yticklabels=['Normal', 'Fraude'])
plt.title('Conjunto de Prueba: Matriz de Confusión (%)')
plt.ylabel('Real')
plt.xlabel('Predicho')

plt.tight_layout()
plt.savefig('final_evaluation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualizaciones de evaluación final guardadas como 'final_evaluation_results.png'")

# Reporte de resumen
print(f"\n{'='*80}")
print("RESUMEN DE EVALUACIÓN FINAL")
print(f"{'='*80}")
print(f"Configuración del Modelo: {mejor_config['nombre']}")
print(f"Capas de Codificación: {mejor_config['capas_codificacion']}")
print(f"Función de Activación: {mejor_config['activacion']}")
print(f"Umbral Óptimo: {umbral_optimo:.6f}")
print(f"")
print(f"Rendimiento en Conjunto de Prueba:")
print(f"  - F1-Score: {f1_prueba:.4f}")
print(f"  - Precisión: {precision_prueba:.4f}")
print(f"  - Recall: {recall_prueba:.4f}")
print(f"  - Exactitud: {exactitud:.4f}")
print(f"  - AUC-ROC: {auc_prueba:.4f}")
print(f"")
print(f"Resultados de Detección de Fraude:")
print(f"  - Total de Transacciones Fraudulentas: {np.sum(y_test == 1)}")
print(f"  - Correctamente Detectadas: {vp}")
print(f"  - Perdidas (Falsos Negativos): {fn}")
print(f"  - Falsas Alarmas: {fp}")
print(f"  - Tasa de Detección: {sensibilidad:.1%}")
print(f"")
print(f"¡Fase 4 completada exitosamente!")

# Guardar resumen para generación de reporte
datos_resumen = {
    'nombre_modelo': mejor_config['nombre'],
    'capas_codificacion': mejor_config['capas_codificacion'],
    'activacion': mejor_config['activacion'],
    'umbral_optimo': umbral_optimo,
    'rendimiento_prueba': {
        'f1_score': f1_prueba,
        'precision': precision_prueba,
        'recall': recall_prueba,
        'exactitud': exactitud,
        'auc_roc': auc_prueba
    },
    'resumen_deteccion_fraude': {
        'total_transacciones_fraude': int(np.sum(y_test == 1)),
        'correctamente_detectadas': int(vp),
        'fraudes_perdidos': int(fn),
        'falsas_alarmas': int(fp),
        'tasa_deteccion': sensibilidad
    }
}

with open('evaluation_summary.json', 'w', encoding='utf-8') as f:
    json.dump(datos_resumen, f, indent=2, ensure_ascii=False)

print("Resumen de evaluación guardado para generación de reporte")

