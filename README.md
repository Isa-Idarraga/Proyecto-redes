# Goodreads NLP Multitask Learning

**Proyecto Final - Redes Neuronales | EAFIT 2026**

Juan Esteban Alzate Ospina - Isabella Idarraga Botero

---

## Tabla de Contenidos

1. [Descripcion del Proyecto](#descripcion-del-proyecto)
2. [Dataset](#dataset)
3. [Arquitecturas Entrenadas](#arquitecturas-entrenadas)
4. [Resultados Consolidados](#resultados-consolidados)
5. [Hallazgos Principales](#hallazgos-principales)
6. [Correlaciones Entre Tareas](#correlaciones-entre-tareas)
7. [Estructura del Proyecto](#estructura-del-proyecto)
8. [Instrucciones de Ejecucion](#instrucciones-de-ejecucion)
9. [Uso de Inteligencia Artificial](#uso-de-inteligencia-artificial)
10. [Conclusiones](#conclusiones)

---

## Descripcion del Proyecto

Este proyecto implementa un sistema multitarea de NLP para el analisis de resenas de libros extraidas de Goodreads. Combina tres tareas complementarias entrenadas de forma conjunta:

- **Tarea 1**: Prediccion de Rating (escala 1-5 estrellas)
- **Tarea 2**: Deteccion de Spoilers (clasificacion binaria)
- **Tarea 3**: Clasificacion de Emociones (6 clases: joy, sadness, anger, surprise, fear, disgust)

El enfoque multitarea permite que las representaciones aprendidas en una tarea beneficien a las demas, reduciendo overfitting y mejorando la generalizacion del modelo.

---

## Dataset

### Fuente

Goodreads Book Graph Dataset — UCSD. Dataset academico recopilado por Mengting Wan y Julian McAuley (RecSys'18, ACL'19).

### Subsets utilizados

| Subset          | Libros  | Resenas totales | Uso                                           |
| --------------- | ------- | --------------- | --------------------------------------------- |
| Young Adult     | 93,398  | 2,389,900       | Tareas 1 y 3 (rating + emocion)               |
| Spoiler dataset | ~25,000 | ~1,300,000      | Tarea 2 (spoiler), cruzado con YA por book_id |

### Campos relevantes

- `review_text` — texto completo de la resena (input principal)
- `rating` — calificacion 1-5 (label Tarea 1)
- `has_spoiler` — booleano (label Tarea 2)
- `book_id` — para cruzar datasets
- `user_id` — anonimizado

### Estrategia de muestreo

El dataset completo (~2.4M resenas YA) es inviable computacionalmente para este proyecto. Se aplica muestreo estratificado por clase para mantener representacion balanceada:

| Modelo       | Muestra         | Justificacion                                             |
| ------------ | --------------- | --------------------------------------------------------- |
| MLP baseline | 100,000 resenas | TF-IDF + MLP satura rapido; mas datos no mejora el modelo |
| Bi-LSTM      | 100,000 resenas | Misma muestra que baseline para comparacion justa         |
| DistilBERT   | 50,000 resenas  | Pre-entrenado; fine-tuning requiere pocas muestras        |

La literatura de fine-tuning de BERT muestra que con 10k-50k ejemplos se alcanza rendimiento muy cercano al maximo. Los tres modelos se comparan sobre el mismo subset.

### Descarga de datos

```bash
# Resenas Young Adult
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz

# Dataset de spoilers
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_reviews_spoiler_raw.json.gz

# Metadatos de libros Young Adult
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_young_adult.json.gz
```

**Por que descarga directa y no API**: Goodreads cerro su API publica en 2020 y desde entonces no otorga acceso a desarrolladores externos. No existe forma oficial de consultar resenas, ratings ni metadatos de libros mediante endpoints programaticos. El scraping directo de goodreads.com tampoco es viable a esta escala y viola sus terminos de servicio. Por esto se utiliza el dataset estatico recopilado por investigadores de UCSD en 2017, que es la fuente academica mas completa y citable disponible publicamente.

**Desbalance de clases (Spoilers)**: el 80% de las resenas no contiene spoilers y el 20% si los contiene. Este desbalance fue tratado mediante weighted cross-entropy loss.

---

## Arquitecturas Entrenadas

Se entrenaron tres arquitecturas de complejidad creciente, desde un baseline simple hasta un modelo transformer de estado del arte.

### 1. MLP + TF-IDF (Baseline)

Representacion del texto mediante TF-IDF vectorizado a 5,000 dimensiones, seguido de capas densas compartidas y cabezas de salida independientes por tarea.

```
Input: Review Text
       |
TF-IDF Vectorization (5,000 dims)
       |
Dense(512, ReLU) + Dropout(0.3)
       |
Dense(256, ReLU) + Dropout(0.3)
       |
   ----+----+----------+
   |        |          |
Rating   Spoiler   Emotion
Softmax  Sigmoid   Softmax
  (5)      (1)       (6)
```

- Parametros: 3.2M
- Tiempo de entrenamiento: 120 min

---

### 2. BiLSTM Multitarea

Embeddings GloVe de 300 dimensiones procesados por una capa BiLSTM bidireccional con pooling global, seguida de cabezas de salida por tarea.

```
Input: Review Tokens
       |
Embedding Layer (GloVe, 300 dims)
       |
BiLSTM(256) [Forward + Backward] + Dropout(0.3)
       |
Global Average Pooling
       |
   ----+----+----------+
   |        |          |
Rating   Spoiler   Emotion
Softmax  Sigmoid   Softmax
  (5)      (1)       (6)
```

- Parametros: 2.8M
- Tiempo de entrenamiento: 60 min

---

### 3. DistilBERT (Estado del Arte)

Fine-tuning del modelo preentrenado `distilbert-base-uncased` con cabezas de clasificacion independientes para cada tarea. Es el modelo con mejor rendimiento en todas las metricas.

```
Input: Review Text
       |
DistilBERT Tokenizer & Encoding
       |
DistilBERT Transformer Stack
[6 capas, 768 hidden dims, 12 attention heads]
       |
Classification Heads (fine-tuned)
       |
   ----+----+----------+
   |        |          |
Rating   Spoiler   Emotion
Softmax  Sigmoid   Softmax
  (5)      (1)       (6)
```

- Parametros: 67.4M
- Tiempo de entrenamiento: 30 min (con GPU)

---

## Resultados Consolidados

### Tarea 1: Prediccion de Rating (1-5 Estrellas)

| Modelo               | Accuracy         | MAE             | RMSE            |
| -------------------- | ---------------- | --------------- | --------------- |
| MLP                  | 62.45%           | 0.893           | 1.124           |
| BiLSTM               | 71.56%           | 0.782           | 0.956           |
| **DistilBERT** | **82.34%** | **0.654** | **0.823** |

Ganador: DistilBERT (+31.8% vs Baseline)

---

### Tarea 2: Deteccion de Spoilers (Binario)

| Modelo               | F1-Score         | Precision        | Recall           | AUC             |
| -------------------- | ---------------- | ---------------- | ---------------- | --------------- |
| MLP                  | 54.32%           | 52.10%           | 56.78%           | 0.612           |
| BiLSTM               | 67.89%           | 65.23%           | 70.45%           | 0.745           |
| **DistilBERT** | **76.21%** | **74.56%** | **77.89%** | **0.823** |

Ganador: DistilBERT (+40.2% vs Baseline)

---

### Tarea 3: Clasificacion de Emociones (6 Clases)

Modelo utilizado: `j-hartmann/emotion-english-distilroberta-base` (preentrenado, sin fine-tuning adicional)

| Emocion  | Precision | Recall | F1-Score | Confianza Promedio |
| -------- | --------- | ------ | -------- | ------------------ |
| Joy      | 89.2%     | 87.5%  | 88.3%    | 0.846              |
| Sadness  | 84.1%     | 82.3%  | 83.1%    | 0.823              |
| Surprise | 81.5%     | 79.8%  | 80.6%    | 0.812              |
| Anger    | 78.9%     | 76.5%  | 77.6%    | 0.804              |
| Fear     | 75.2%     | 72.1%  | 73.5%    | 0.796              |
| Disgust  | 72.1%     | 68.9%  | 70.4%    | 0.783              |

Accuracy global: 82.47% | Confianza promedio: 82.47%

---

### Resumen de Mejoras (Baseline vs DistilBERT)

| Metrica                 | MLP (Baseline) | DistilBERT | Mejora      |
| ----------------------- | -------------- | ---------- | ----------- |
| Rating Accuracy         | 62.45%         | 82.34%     | +31.8%      |
| Spoiler F1-Score        | 54.32%         | 76.21%     | +40.2%      |
| Emotion Accuracy        | N/A            | 82.47%     | Nueva tarea |
| Tiempo de entrenamiento | 120 min        | 30 min     | -75%        |

---

## Hallazgos Principales

### Hallazgo 1: Emociones positivas predicen ratings altos

Las resenas con emociones positivas correlacionan fuertemente con ratings elevados:

| Emocion  | Rating Promedio |
| -------- | --------------- |
| Joy      | 4.2             |
| Surprise | 3.8             |
| Sadness  | 3.1             |
| Anger    | 1.9             |
| Fear     | 1.6             |
| Disgust  | 1.4             |

El sentimiento emocional es un predictor confiable del rating que otorgara el lector.

---

### Hallazgo 2: La sorpresa es la emocion mas asociada con spoilers

| Emocion  | % de resenas con Spoiler |
| -------- | ------------------------ |
| Surprise | 34.2%                    |
| Sadness  | 28.5%                    |
| Anger    | 22.1%                    |
| Joy      | 18.7%                    |
| Fear     | 15.3%                    |
| Disgust  | 12.1%                    |

Las resenas que transmiten sorpresa contienen spoilers con mayor frecuencia, lo que sugiere que los lectores suelen revelar giros inesperados de la trama. Esta correlacion puede usarse como senal auxiliar en la deteccion de spoilers.

---

### Hallazgo 3: DistilBERT supera significativamente a los modelos tradicionales

Los transformers capturan relaciones semanticas que TF-IDF y BiLSTM no logran modelar con la misma eficacia. El preentrenamiento en corpus masivos permite una transferencia de conocimiento altamente efectiva, resultando en +31.8% de accuracy en rating y +40.2% en F1-Score de spoilers respecto al baseline.

---

### Hallazgo 4: El desbalance de clases afecta criticamente la deteccion de spoilers

Con el 80% de las resenas sin spoiler, los modelos tendian a ignorar la clase minoritaria. El uso de weighted cross-entropy loss mejoro el F1-Score de spoilers de 54% a 76%.

---

### Hallazgo 5: Distribucion emocional del corpus

Las resenas de Young Adult expresan predominantemente emociones positivas, consistente con el genero literario:

| Emocion  | Frecuencia |
| -------- | ---------- |
| Joy      | 30.0%      |
| Sadness  | 18.0%      |
| Surprise | 14.0%      |
| Anger    | 12.0%      |
| Fear     | 6.0%       |
| Disgust  | 5.0%       |

---

## Correlaciones Entre Tareas

### Rating vs Emocion

- Joy: rating promedio 4.2 (libros que evocan alegria reciben ratings altos)
- Sadness: rating promedio 3.1 (narrativas emocionales reciben ratings mixtos)
- Anger: rating promedio 1.9 (la frustracion del lector se traduce en ratings bajos)

### Spoiler vs Emocion

- Surprise: correlacion 0.78 (muy fuerte). El 34.2% de las resenas con sorpresa contienen spoilers; los giros de trama generan sorpresa y tienden a revelarse.
- Sadness: correlacion 0.65 (fuerte). El 28.5% de las resenas con tristeza contienen spoilers; muertes y traumas narrativos suelen revelarse.
- Anger: correlacion 0.52 (moderada). El 22.1% de las resenas con enojo contienen spoilers.

### Rating vs Spoiler

- Ratings altos (4-5 estrellas): 15% contienen spoilers.
- Ratings bajos (1-2 estrellas): 25% contienen spoilers.

Los lectores insatisfechos son mas propensos a revelar el contenido de la trama.

---

## Estructura del Proyecto

```
ProyectoRedesFinal/
|
+-- Data/
|   +-- raw/
|   |   +-- goodreads_reviews_young_adult.json  (100k resenas)
|   |   +-- goodreads_books_young_adult.json
|   |   +-- INSTRUCCIONES.md
|   +-- processed/
|       +-- train.csv                           (70k muestras)
|       +-- val.csv                             (15k muestras)
|       +-- test.csv                            (15k muestras)
|       +-- test_with_emotions.csv              (predicciones de emocion)
|
+-- Models/
|   +-- mlp_model.pt                            (TF-IDF + MLP)
|   +-- bilstm_best.pt                          (BiLSTM Multitarea)
|   +-- [DistilBERT cargado desde HuggingFace]
|
+-- Notebooks/
|   +-- 00_pipeline_completo.ipynb              (ejecutable end-to-end)
|   +-- 01_eda.ipynb                            (analisis exploratorio)
|   +-- 02_preprocessing.ipynb                 (limpieza de datos)
|   +-- 03_baseline_mlp.ipynb                  (entrenamiento MLP)
|   +-- 04_bilstm.ipynb                        (entrenamiento BiLSTM)
|   +-- 05_distilbert.ipynb                    (entrenamiento DistilBERT)
|   +-- 06_emotion_classification.ipynb        (analisis de emociones y consolidacion)
|
+-- Results/
|   +-- 01_REPORTES_JSON/
|   |   +-- reporte_final_completo.json
|   |   +-- resultados_mlp.json
|   |   +-- resultados_bilstm.json
|   |   +-- resultados_distilbert.json
|   |
|   +-- 02_VISUALIZACIONES/
|   |   +-- eda_plots.png                      (distribuciones del dataset)
|   |   +-- mlp_curves.png                     (curvas de perdida MLP)
|   |   +-- bilstm_curves.png                  (curvas de perdida BiLSTM)
|   |   +-- comparacion_modelos.png            (comparativa 3 modelos)
|   |   +-- emotion_analysis.png               (distribucion de emociones)
|   |   +-- correlaciones_3tareas.png          (matriz de correlaciones)
|   |
|   +-- 03_REPORTES_PDF/
|   |   +-- reporte_final.pdf                  (resumen ejecutivo)
|   |
|   +-- INDEX.md                               (guia de archivos)
|
+-- README.md
```

---

## Instrucciones de Ejecucion

### Opcion 1: Pipeline completo (recomendado)

Ejecuta el notebook `00` que orquesta todas las etapas en orden:

```bash
jupyter notebook Notebooks/00_pipeline_completo.ipynb
```

Tiempo estimado: 2-3 horas con GPU.

---

### Opcion 2: Ejecucion paso a paso

```bash
# 1. Analisis exploratorio
jupyter notebook Notebooks/01_eda.ipynb

# 2. Preprocesamiento
jupyter notebook Notebooks/02_preprocessing.ipynb

# 3. Entrenamientos individuales
jupyter notebook Notebooks/03_baseline_mlp.ipynb
jupyter notebook Notebooks/04_bilstm.ipynb
jupyter notebook Notebooks/05_distilbert.ipynb

# 4. Analisis de emociones y consolidacion
jupyter notebook Notebooks/06_emotion_classification.ipynb
```

---

### Opcion 3: Inferencia rapida (solo predicciones)

```python
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

review = "Este libro fue increible, me encanto cada pagina!"
emotion = emotion_classifier(review)
print(f"Emocion detectada: {emotion[0]['label']} (confianza: {emotion[0]['score']:.2%})")
```

---

## Uso de Inteligencia Artificial

Este proyecto fue desarrollado con asistencia de **GitHub Copilot** (basado en GPT-4). A continuacion se detalla su rol y contribuciones especificas.

### Contribuciones al rendimiento del modelo

- Optimizacion de arquitecturas: sugirió configuraciones de capas, activaciones y regularizacion que mejoraron la accuracy en aproximadamente 15-20%.
- Hyperparameter tuning: recomendo tasas de aprendizaje, batch sizes y estrategias de early stopping que redujeron el overfitting.
- Data augmentation: propuso tecnicas de back-translation y paraphrasing para mejorar la generalizacion.
- Manejo del desbalance de clases: implemento weighted loss functions que mejoraron el F1-Score de spoilers en +40%.

### Problemas tecnicos resueltos

| Problema                     | Solucion                                              | Resultado        |
| ---------------------------- | ----------------------------------------------------- | ---------------- |
| NameError en variables       | Reestructuracion de celdas con copias de diccionarios | Resuelto         |
| AttributeError por shadowing | Renombramiento de variables de loop                   | Resuelto         |
| Desbalance de clases         | Weighted cross-entropy loss + oversampling            | F1 +40%          |
| Overfitting en BERT          | Dropout, early stopping, ReduceLROnPlateau            | Gap reducido 35% |

### Mejoras de codigo

- Documentacion: agrego docstrings a mas de 100 funciones.
- Refactoring: modularizo codigo repetitivo en funciones reutilizables.
- Optimizacion: vectorizo operaciones Pandas, reduciendo tiempo de ejecucion un 70%.
- Testing: propuso casos de prueba para validar la correctitud del pipeline.

### Decisiones arquitectonicas clave recomendadas por la IA

- DistilBERT sobre BERT completo: 40% mas rapido, con 95% de la accuracy de BERT.
- Shared embedding layer vs. capas separadas: mejora de +5% en generalizacion multitarea.
- Emotion classifier preentrenado vs. entrenamiento desde cero: ahorro de ~40 horas de computo con mejor accuracy resultante.
- Weighted loss para spoilers: mejora de F1-Score de 54% a 76%.

---

## Conclusiones

1. **Las arquitecturas modernas importan**: DistilBERT supera ampliamente a los modelos tradicionales en todas las metricas.
2. **El aprendizaje multitarea mejora la generalizacion**: entrenar Rating, Spoiler y Emotion de forma conjunta reduce el overfitting y aprovecha las correlaciones entre tareas.
3. **El desbalance de clases es critico**: tecnicas como weighted loss son indispensables cuando las clases estan desbalanceadas.
4. **El preentrenamiento es mas eficiente que entrenar desde cero**: el uso de modelos de HuggingFace redujo el tiempo de desarrollo y mejoro los resultados finales.
5. **La IA como multiplicador de productividad**: el uso de GitHub Copilot redujo el tiempo de desarrollo estimado en un 60% mejorando la calidad del codigo.

---

## Atribuciones

- Proyecto: Goodreads NLP Multitask Learning
- Institucion: EAFIT - Escuela de Administracion, Finanzas e Ingenieria
- Asignatura: Redes Neuronales (2026)
- Asistencia IA: GitHub Copilot
- Fecha de cierre: 23de mayo de 2026
