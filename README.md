Detector de Fraude con Autoencoders

Muchaches se adjunta resultados de cada punto solicitado junton con los artefactos correspondientes.

Resultados preliminares:

🎯 F1-Score: 35.2% 

🔍 Precisión: 23.8% (De cada 4 alertas, 1 es fraude real)

🎣 Recall: 67.3% 

🏆 AUC-ROC: 94.4% 

Plain Text

📁 deteccion-fraude-autoencoders/
├── 📂 src/                    # Python
├── 📂 docs/                   # Documentación técnica completa
├── 📂 models/                 # Los 3 autoencoders entrenados
├── 📂 results/                # Resultados y métricas en JSON
├── 📂 visualizations/         # Gráficos que están buenos
├── 📄 README.md              # Este archivo 
├── 📄 requirements.txt       # Las librerías que necesitan
└── 📄 LICENSE               # MIT License (úsenlo tranquilos)



Paso 1: Clonar

Bash

git clone https://github.com/fescudero/deteccion-fraude-autoencoders.git
cd deteccion-fraude-autoencoders


Paso 2: Instalar

Bash


pip install -r requirements.txt


Paso 3: Datasexxx


Opción 1 

1.
Vayan a Kaggle - Credit Card Fraud Detection

2.
Descarguen el archivo creditcard.csv

3.
Pónganlo en la carpeta data/ del proyecto

Opción 2 


# Instalar kaggle 
pip install kaggle

# API key de Kaggle 
# Después ejecuten:
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
mv creditcard.csv data/


Paso 4: 

Para entrenar todo desde cero:

Bash

# 1. Exploración de datos
python src/vae_fraud_detection.py

# 2. Entrenar los 3 autoencoders
python src/autoencoder_simple.py

# 3. Optimizar el umbral
python src/threshold_optimization.py

# 4. Evaluación final
python src/final_evaluation.py

Si solo quieren ver los resultados

Bash

python src/final_evaluation.py

1.
Entrenamiento: Le mostramos al autoencoder SOLO transacciones normales (como enseñarle a alguien qué es "normal")

2.
Detección: Cuando ve una transacción nueva, trata de "reconstruirla"

3.
Si no puede reconstruirla bien (error alto), probablemente sea fraude

4.
Umbral óptimo: Encontramos el punto donde funciona mejor

📊 Los resultados.

Configuraciones probadas:

•
Config 1 (Pequeño): [64, 32, 16] neuronas

•
Config 2 (Mediano): [32, 16, 8] neuronas ⭐ Este ganó

•
Config 3 (Grande): [128, 64, 32] neuronas

El Config 2 encontró el equilibrio justo entre aprender bien y no sobreajustarse.

📈 Visualizaciones

Aqupi van a encontrar:

•
📊 Comparación de entrenamiento de los 3 modelos

•
🎯 Optimización del umbral (curva ROC incluida)

•
🔍 Matriz de confusión final

•
📈 Exploración inicial de los datos

📚 Documentación técnica


docs/Reporte_Deteccion_Fraude_VAE.md - Reporte completo en español

•
docs/VAE_Fraud_Detection_Report.md - Versión en inglés (legacy)

📝 Licencia

MIT License - Básicamente pueden hacer lo que quieran con esto
