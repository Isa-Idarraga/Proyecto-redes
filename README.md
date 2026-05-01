# Goodreads NLP Multitask

Predicción de rating, detección de spoilers y clasificación de emociones en reseñas de libros del género Young Adult — Universidad EAFIT, Redes Neuronales 2025

---

## Problema a resolver

Las plataformas de reseñas literarias como Goodreads generan millones de textos con información valiosa pero dispersa. Este proyecto aborda tres tareas de NLP simultáneas sobre el mismo corpus de reseñas del género Young Adult:

| # | Tarea | Tipo | Output |
|---|-------|------|--------|
| 1 | Predicción de rating | Clasificación multiclase | 1–5 estrellas |
| 2 | Detección de spoilers | Clasificación binaria | sí / no |
| 3 | Clasificación de emoción | Clasificación multiclase | alegría, tristeza, enojo, sorpresa, miedo, disgusto |

La tarea de predicción de rating va más allá del análisis de sentimiento binario — el modelo debe captar matices como la diferencia entre una reseña de 3 y una de 4 estrellas. La detección de spoilers tiene alto impacto práctico para plataformas de contenido. Las etiquetas de emoción se generan con un modelo pre-entrenado (`j-hartmann/emotion-english-distilroberta-base`) como paso de preprocesamiento, dado que el dataset no incluye esta anotación.

---

## Datos

**Fuente:** [Goodreads Book Graph Dataset — UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)  
Dataset académico recopilado por Mengting Wan y Julian McAuley (RecSys'18, ACL'19).

### Subsets utilizados

| Subset | Libros | Reseñas totales | Uso |
|--------|--------|-----------------|-----|
| Young Adult | 93,398 | 2,389,900 | Tareas 1 y 3 (rating + emoción) |
| Spoiler dataset | ~25,000 | ~1,300,000 | Tarea 2 (spoiler), cruzado con YA por `book_id` |

### Campos relevantes

- `review_text` — texto completo de la reseña *(input principal)*
- `rating` — calificación 1–5 *(label Tarea 1)*
- `has_spoiler` — booleano *(label Tarea 2)*
- `book_id` — para cruzar datasets
- `user_id` — anonimizado

### Estrategia de muestreo

El dataset completo (~2.4M reseñas YA) es inviable computacionalmente para este proyecto. Se aplica muestreo estratificado por clase para mantener representación balanceada:

| Modelo | Muestra | Justificación |
|--------|---------|---------------|
| MLP baseline | 100,000 reseñas | TF-IDF + MLP satura rápido; más datos no mejora el modelo |
| Bi-LSTM | 100,000 reseñas | Misma muestra que baseline para comparación justa |
| DistilBERT | 50,000 reseñas | Pre-entrenado; fine-tuning requiere pocas muestras |

> La literatura de fine-tuning de BERT muestra que con 10k–50k ejemplos se alcanza rendimiento muy cercano al máximo. Los tres modelos se comparan sobre el mismo subset.

### Descarga de datos

```bash
# Young Adult reviews
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz

# Spoiler dataset
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_reviews_spoiler_raw.json.gz
```

> **Por qué descarga directa y no API:** Goodreads cerró su API pública en 2020 y desde entonces no otorga acceso a desarrolladores externos. No existe forma oficial de consultar reseñas, ratings ni metadatos de libros mediante endpoints programáticos. El scraping directo de goodreads.com tampoco es viable a esta escala y viola sus términos de servicio. Por esto se utiliza el dataset estático recopilado por investigadores de UCSD en 2017, que es la fuente académica más completa y citable disponible públicamente.

---

## Arquitecturas

### Arquitectura base — MLP + TF-IDF

Línea de referencia (baseline). Representación del texto con TF-IDF (20k tokens), red densa simple con una cabeza de salida por tarea.

```
TF-IDF (20,000 dim)
    → Dense(256) + ReLU + Dropout(0.3)
    → Dense(128) + ReLU + Dropout(0.3)
    → [Cabeza rating]   Dense(5)  + Softmax
    → [Cabeza spoiler]  Dense(1)  + Sigmoid
    → [Cabeza emocion]  Dense(6)  + Softmax
```

Métricas: Accuracy, MAE (rating), F1-score (spoiler y emoción)

---

### Arquitectura propuesta — Bi-LSTM multitarea

Aprovecha la naturaleza secuencial del texto. Encoder compartido entre las tres tareas (aprendizaje multitarea), con cabezas específicas por tarea.

```
Embedding (vocab x 128, entrenable)
    → Bidirectional LSTM (128 unidades)
    → Bidirectional LSTM (64 unidades)
    → Dropout(0.4)
    → [Cabeza rating]   Dense(64) + ReLU → Dense(5)  + Softmax
    → [Cabeza spoiler]  Dense(32) + ReLU → Dense(1)  + Sigmoid
    → [Cabeza emocion]  Dense(64) + ReLU → Dense(6)  + Softmax
```

Loss total: `a·Loss_rating + b·Loss_spoiler + c·Loss_emocion`  
Los pesos a, b, c se ajustan como hiperparámetros.

---

### Arquitectura con transfer learning — DistilBERT fine-tuning

Usa `distilbert-base-uncased` (66M parámetros, pre-entrenado en Wikipedia + BookCorpus) como encoder. Fine-tuning en dos fases:

- Fase 1 — Feature extraction: pesos de DistilBERT congelados, solo se entrenan las cabezas (3–5 épocas)
- Fase 2 — Fine-tuning: se descongelan las últimas 2 capas con lr=2e-5

```
DistilBERT encoder → [CLS] token (768 dim)
    → Dropout(0.1)
    → [Cabeza rating]   Linear(768→256) + ReLU → Linear(256→5)
    → [Cabeza spoiler]  Linear(768→64)  + ReLU → Linear(64→1)
    → [Cabeza emocion]  Linear(768→256) + ReLU → Linear(256→6)
```

DistilBERT fue pre-entrenado en BookCorpus (texto literario en inglés), lo que lo hace especialmente adecuado para este dominio.

---

## Estructura del repositorio

```
goodreads-nlp-multitask/
├── data/
│   ├── download.sh              # descarga de datasets desde UCSD
│   └── preprocessing.py         # limpieza, muestreo, emotion labeling
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_mlp.ipynb
│   ├── 03_bilstm.ipynb
│   └── 04_distilbert.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── bilstm.py
│   │   └── distilbert.py
│   └── train.py
├── results/
├── requirements.txt
└── README.md
```

---

## Stack tecnológico

- Python 3.10+
- TensorFlow / Keras — MLP y Bi-LSTM
- HuggingFace Transformers — DistilBERT
- HuggingFace Datasets — carga y procesamiento
- scikit-learn — TF-IDF, métricas
- pandas — muestreo estratificado
- `j-hartmann/emotion-english-distilroberta-base` — generación de labels de emoción

---

## Posibles extensiones

- **Incorporar metadatos del libro:** el dataset incluye un subset de metadatos de libros YA (`goodreads_books_young_adult.json.gz`) con campos como género, número de páginas, autor y descripción. Cruzando por `book_id` estos podrían usarse como features adicionales al texto de la reseña, enriqueciendo la representación del input.
- **Modelo multitarea con atención:** agregar un mecanismo de atención sobre el encoder Bi-LSTM para identificar qué partes del texto son más relevantes por tarea.
- **Análisis por género literario:** extender el corpus a otros géneros del dataset (Romance, Fantasy, Mystery) y comparar si los modelos generalizan entre géneros o requieren fine-tuning específico.
- **Detección de spoilers a nivel de oración:** el dataset de spoilers incluye una versión anotada a nivel de oración (`goodreads_reviews_spoiler.json.gz`) que permitiría pasar de clasificación de reseña completa a clasificación de oraciones individuales.

---

## Referencias

- Mengting Wan, Julian McAuley. *Item Recommendation on Monotonic Behavior Chains*. RecSys 2018.
- Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley. *Fine-Grained Spoiler Detection from Large-Scale Review Corpora*. ACL 2019.
- Sanh et al. *DistilBERT, a distilled version of BERT*. 2019.
