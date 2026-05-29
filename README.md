# Goodreads NLP Multitask Learning

**Proyecto Final — Redes Neuronales | EAFIT 2026**  
Juan Esteban Alzate Ospina · Isabella Idarraga Botero

---

## Descripción del Proyecto

Sistema multitarea de NLP para el análisis de reseñas de libros extraídas de Goodreads (género Young Adult). Combina tres tareas sobre el mismo corpus:

| # | Tarea | Tipo | Output |
|---|-------|------|--------|
| 1 | Predicción de rating | Clasificación multiclase | 1–5 estrellas |
| 2 | Detección de spoilers | Clasificación binaria | sí / no |
| 3 | Clasificación de emociones | Inferencia con modelo pre-entrenado | joy, sadness, anger, surprise, fear, disgust, neutral |

Las tareas 1 y 2 se entrenan de forma conjunta (multitarea) en los tres modelos. La tarea 3 se resuelve con inferencia directa del modelo pre-entrenado `j-hartmann/emotion-english-distilroberta-base` sobre el test set — no requiere entrenamiento adicional.

---

## Dataset

**Fuente:** [Goodreads Book Graph Dataset — UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)  
Recopilado por Mengting Wan y Julian McAuley (RecSys'18, ACL'19).

| Subset | Libros | Reseñas totales | Uso |
|--------|--------|-----------------|-----|
| Young Adult | 93,398 | 2,389,900 | Tareas 1 y 3 (rating + emoción) |
| Spoiler dataset | ~25,000 | ~1,300,000 | Tarea 2 (spoiler), cruzado con YA por `review_id` |

**Campos relevantes:**
- `review_text` — texto completo de la reseña (input principal)
- `rating` — calificación 1–5 (label Tarea 1)
- `has_spoiler` — booleano extraído de etiquetas en el texto (label Tarea 2)
- `book_id` / `review_id` — para cruzar datasets

### Estrategia de muestreo

| Modelo | Muestra | Justificación |
|--------|---------|---------------|
| MLP baseline | 100,000 reseñas | TF-IDF + MLP satura rápido; más datos no mejora |
| Bi-LSTM | 100,000 reseñas | Misma muestra que baseline para comparación justa |
| DistilBERT | 50,000 reseñas | Pre-entrenado; fine-tuning requiere pocas muestras |

Split estratificado por rating: **70% train · 15% val · 15% test**

### Descarga de datos

```bash
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_reviews_spoiler_raw.json.gz
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_young_adult.json.gz
```

> Goodreads cerró su API pública en 2020. Se utiliza el dataset estático de UCSD (2017), la fuente académica más completa disponible públicamente.

---

## Arquitecturas

Los tres modelos tienen cabezas de salida para **Tarea 1 (rating)** y **Tarea 2 (spoiler)**. La Tarea 3 (emoción) es independiente y se resuelve con el modelo pre-entrenado en el notebook 06.

### 1. MLP + TF-IDF (Baseline)

```
TF-IDF (20,000 tokens)
    → Dense(256) + ReLU + Dropout(0.3)
    → Dense(128) + ReLU + Dropout(0.3)
    → [Rating]   Dense(5)  + Softmax
    → [Spoiler]  Dense(1)  + Sigmoid
```

Loss total: `0.7 · CrossEntropy(rating) + 0.3 · BCEWithLogits(spoiler)`

### 2. BiLSTM Multitarea

```
Embedding entrenable (20,000 × 128d)
    → BiLSTM(128 unidades) + Dropout(0.4)
    → BiLSTM(64 unidades)
    → último estado oculto
    → [Rating]   Dense(64) + ReLU → Dense(5)  + Softmax
    → [Spoiler]  Dense(32) + ReLU → Dense(1)  + Sigmoid
```

Loss total: `0.7 · CrossEntropy(rating) + 0.3 · BCEWithLogits(spoiler)`  
Early stopping (patience=3), gradient clipping (max_norm=1.0)

### 3. DistilBERT Fine-tuning

```
distilbert-base-uncased (66M parámetros, pre-entrenado Wikipedia + BookCorpus)
    → token [CLS] (768 dim) + Dropout(0.1)
    → [Rating]   Linear(768→256) + ReLU → Linear(256→5)
    → [Spoiler]  Linear(768→64)  + ReLU → Linear(64→1)
```

Fase 1 — Feature extraction: pesos de DistilBERT congelados, solo cabezas (3 épocas, lr=1e-3)  
Fase 2 — Fine-tuning: últimas 2 capas descongeladas (5 épocas, lr=2e-5)  
Max tokens: 128

### Hiperparámetros de entrenamiento

| Parámetro | MLP | BiLSTM | DistilBERT |
|-----------|-----|--------|------------|
| Batch size | 256 | 64 | 32 |
| Épocas | 10 | 10 (ES=3) | 3 + 5 (2 fases) |
| Optimizer | Adam lr=1e-3 | Adam lr=1e-3 | Adam lr=2e-5 |
| Regularización | Dropout 0.3 | Dropout 0.4 + grad clip | Dropout 0.1 |
| Hardware | Apple MPS | Apple MPS | Apple MPS |

---

## Resultados

### Tareas 1 y 2 — Rating y Spoiler

| Modelo | Acc Rating | MAE Rating | F1 Spoiler |
|--------|------------|------------|------------|
| MLP + TF-IDF | 45.7% | 0.747 | 0.078 |
| BiLSTM Multitarea | 48.6% | 0.672 | 0.000 |
| **DistilBERT** | **54.9%** | **0.601** | 0.000 |

> **Spoiler F1 = 0 en BiLSTM y DistilBERT:** el dataset cruzado YA × spoilers resultó con un desbalance del 94.1% (sin spoiler) vs 5.9% (con spoiler). Ambos modelos colapsaron a predecir siempre la clase mayoritaria. El MLP capturó parcialmente la clase minoritaria (F1=0.078).

### Tarea 3 — Clasificación de emociones

Inferencia con `j-hartmann/emotion-english-distilroberta-base` sobre 15,000 reseñas del test set.  
Confianza promedio: **69.7%**

| Emoción | Reseñas | % | Confianza promedio |
|---------|---------|---|--------------------|
| Neutral | 3,818 | 25.5% | 62.8% |
| Joy | 3,791 | 25.3% | 78.3% |
| Sadness | 2,079 | 13.9% | 70.5% |
| Surprise | 1,818 | 12.1% | 71.3% |
| Disgust | 1,808 | 12.1% | 65.5% |
| Fear | 954 | 6.4% | 70.6% |
| Anger | 732 | 4.9% | 63.6% |

---

## Correlaciones entre Tareas

El notebook `06_emotion_classification.ipynb` calcula y visualiza las relaciones entre las tres tareas sobre el test set (ver `Results/02_VISUALIZACIONES/`):

- **Rating vs Emoción:** emociones positivas (joy, surprise) se asocian a ratings altos; negativas (anger, disgust) a ratings bajos.
- **Spoiler vs Emoción:** surprise es la emoción con mayor porcentaje de reseñas con spoiler — los giros de trama tienden a revelarse.
- **Rating vs Spoiler:** las reseñas con ratings bajos (1–2 estrellas) tienen mayor proporción de spoilers que las de ratings altos.

---

## Estructura del Proyecto

```
ProyectoRedesFinal/
├── Data/
│   ├── raw/
│   │   └── INSTRUCCIONES.md
│   └── processed/
│       ├── train.csv                  (70,000 muestras)
│       ├── val.csv                    (15,000 muestras)
│       ├── test.csv                   (15,000 muestras)
│       └── test_with_emotions.csv     (test + predicciones de emoción)
├── Models/
│   ├── mlp_model.pt
│   ├── bilstm_best.pt
│   └── [DistilBERT se carga desde HuggingFace]
├── Notebooks/
│   ├── 00_pipeline_completo.ipynb     (pipeline end-to-end ejecutable)
│   ├── 01_eda.ipynb                   (análisis exploratorio)
│   ├── 02_preprocessing.ipynb         (limpieza y muestreo)
│   ├── 03_baseline_mlp.ipynb          (MLP + TF-IDF)
│   ├── 04_bilstm.ipynb                (BiLSTM multitarea)
│   ├── 05_distilbert.ipynb            (DistilBERT fine-tuning + reporte PDF)
│   └── 06_emotion_classification.ipynb (emociones + correlaciones + consolidación)
└── Results/
    ├── 01_REPORTES_JSON/
    │   ├── reporte_final_completo.json
    │   ├── resultados_mlp.json
    │   ├── resultados_bilstm.json
    │   └── resultados_distilbert.json
    ├── 02_VISUALIZACIONES/
    │   ├── eda_plots.png
    │   ├── mlp_curves.png
    │   ├── bilstm_curves.png
    │   ├── comparacion_modelos.png
    │   ├── emotion_analysis.png
    │   └── correlaciones_3tareas.png
    └── 03_REPORTES_PDF/
        └── reporte_final.pdf
```

---

## Instrucciones de Ejecución

**Opción 1 — Pipeline completo (recomendado)**

```bash
jupyter notebook Notebooks/00_pipeline_completo.ipynb
```

Tiempo estimado: 2–4 horas con GPU.

**Opción 2 — Paso a paso**

```bash
jupyter notebook Notebooks/01_eda.ipynb
jupyter notebook Notebooks/02_preprocessing.ipynb
jupyter notebook Notebooks/03_baseline_mlp.ipynb
jupyter notebook Notebooks/04_bilstm.ipynb
jupyter notebook Notebooks/05_distilbert.ipynb
jupyter notebook Notebooks/06_emotion_classification.ipynb
```

---

## Stack Tecnológico

- Python 3.10+
- PyTorch — MLP y BiLSTM
- HuggingFace Transformers — DistilBERT y clasificador de emociones
- scikit-learn — TF-IDF, métricas
- pandas / numpy — procesamiento de datos
- matplotlib / seaborn — visualizaciones
- reportlab — generación del PDF

---

## Uso de Inteligencia Artificial
 
Este proyecto fue desarrollado por el equipo con asistencia de **Claude (Anthropic)** como herramienta de apoyo durante el proceso.
 
El equipo fue responsable de diseñar la solución end-to-end: definir las tres tareas, seleccionar y justificar los datasets, plantear las arquitecturas progresivas (MLP → BiLSTM → DistilBERT), implementar y ejecutar todos los notebooks, interpretar los resultados y redactar las conclusiones. Las decisiones técnicas clave — como el uso de aprendizaje multitarea con encoder compartido, el muestreo estratificado, el manejo del desbalance de clases y la estrategia de fine-tuning en dos fases — fueron tomadas y validadas por el equipo.
 
Claude se utilizó como asistente de programación: apoyó en la escritura del código de los notebooks (arquitecturas, loops de entrenamiento, preprocesamiento), depuró errores puntuales (e.g. incompatibilidad de `ReduceLROnPlateau` con la versión de PyTorch, `NameError` por reinicio de kernel), y apoyó en la redacción y estructura de la documentación. Todo el código fue entendido, revisado y ejecutado por el equipo.

---

## Dependencias
 
Probado con Python 3.10+. Instalar con:
 
```bash
pip install torch torchvision transformers datasets scikit-learn pandas numpy matplotlib seaborn reportlab
```
 
| Librería | Uso |
|----------|-----|
| `torch` | MLP y BiLSTM |
| `transformers` | DistilBERT y clasificador de emociones |
| `datasets` | Carga de datos HuggingFace |
| `scikit-learn` | TF-IDF, métricas |
| `pandas` / `numpy` | Procesamiento de datos |
| `matplotlib` / `seaborn` | Visualizaciones |
| `reportlab` | Generación del PDF |

---

## Conclusiones

1. **DistilBERT supera a los modelos tradicionales** en predicción de rating (54.9% vs 45.7% del MLP) con menor tiempo de entrenamiento.
2. **La detección de spoilers es el desafío principal:** el desbalance real del dataset cruzado (94%/6%) fue más severo de lo estimado, colapsando el F1 a cero en BiLSTM y DistilBERT.
3. **El clasificador de emociones pre-entrenado funcionó bien** con confianza promedio del 69.7% sin fine-tuning adicional.
4. **El aprendizaje multitarea es viable** con encoder compartido, aunque el desbalance extremo de spoilers interfiere con el aprendizaje de las demás tareas.

---

## Referencias

- Wan, M., McAuley, J. (2018). *Item Recommendation on Monotonic Behavior Chains*. RecSys.
- Wan, M. et al. (2019). *Fine-Grained Spoiler Detection from Large-Scale Review Corpora*. ACL.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT*. NeurIPS Workshop.
- Hartmann, J. (2022). *emotion-english-distilroberta-base*. HuggingFace.
