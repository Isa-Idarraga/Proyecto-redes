# 📥 Descargar Datos — Goodreads NLP Multitask

Los datos de este proyecto provienen del **Goodreads Book Graph Dataset** de la Universidad de San Diego (UCSD).

## Archivos requeridos

Coloca estos 3 archivos en esta carpeta (`Data/raw/`):

| Archivo | Tamaño | URL |
|---------|--------|-----|
| `goodreads_reviews_young_adult.json` | 2.5 GB | [Descargar](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) |
| `goodreads_reviews_spoiler_raw.json` | 5.3 GB | [Descargar](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) |
| `goodreads_books_young_adult.json` | 244 MB | [Descargar](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) |

## Pasos

1. **Visita el sitio oficial:**
   https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

2. **Busca en la sección "Goodreads Book Graph" los tres archivos arriba**

3. **Descárgalos y extrae el contenido en esta carpeta:**
   ```
   ProyectoRedesFinal/
   └── Data/
       └── raw/
           ├── goodreads_reviews_young_adult.json
           ├── goodreads_reviews_spoiler_raw.json
           └── goodreads_books_young_adult.json
   ```

4. **Ejecuta el notebook:**
   ```
   Notebooks/01_eda.ipynb
   ```

## ⚠️ Nota sobre tamaño

Los archivos son **bastante grandes** (~7.5 GB en total). 
- Carga de discos duros: ~15-30 minutos
- Requiere al menos 16 GB RAM para el procesamiento

## Fuente

**Mengtng Wan & Julian McAuley** (2018-2019)
- Paper: "Item Recommendation on Monotonic Behavior Chains" (RecSys 2018)
- Paper: "A Large-Scale Heterogeneous Graph Benchmark" (ACL 2019)

---

Una vez colocados los archivos aquí, todos los notebooks funcionarán automáticamente. ✅
