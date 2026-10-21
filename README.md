# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** Juan David Sierra Orozco
- **Código estudiantil:** 1000027039
- **Usuario de GitHub:** [jdj40211](https://github.com/jdj40211)
- **Fecha:** 21 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#bloque-0--teoría) |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#bloque-0--teoría) |
| T3 | Inserción frente a búsqueda | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#bloque-0--teoría) |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#bloque-0--teoría) |
| T5 | El costo de la fuerza bruta | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#bloque-0--teoría) |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ✅ | [A1_gradio.png](evidencias/A1_gradio.png), [notebook 02b](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) |
| A2 | Tu propio corpus de documentos | 6 | ✅ | [A2_corpus.png](evidencias/A2_corpus.png), [corpus](data/company_documents.json) |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | [notebook 02b](notebooks/02b_rag_app_nvidia_nim_colab.ipynb), [RESPUESTAS.md](RESPUESTAS.md) |
| A4 | Muestra el score de similitud en las fuentes | 5 | ✅ | [A4_scores.png](evidencias/A4_scores.png) |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ⬜ |  |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ⬜ |  |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ⬜ |  |
| B4 | Preguntas en español sobre papers en inglés | 5 | ⬜ |  |
| B5 | Escribe tu propio quiz con ground truth | 5 | ⬜ |  |
| R1 | Reflexión final con tus datos | 10 | ⬜ |  |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | [gráficas](evidencias/C1_recall_efsearch.png), [04_challenges](notebooks/04_challenges.ipynb) |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ✅ | [04_challenges](notebooks/04_challenges.ipynb) |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ✅ | [C3_gradio.png](evidencias/C3_gradio.png), [04_challenges](notebooks/04_challenges.ipynb) |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ✅ | [eval/preguntas.csv](eval/preguntas.csv), [04_challenges](notebooks/04_challenges.ipynb) |
| C5 | MMR frente a similitud pura (reto) | 8 | ✅ | [04_challenges](notebooks/04_challenges.ipynb) |
| C6 | Persistencia del índice en disco (reto) | 8 | ✅ | [04_challenges](notebooks/04_challenges.ipynb) |

**Autoevaluación:** 60 / 100 puntos (núcleo 30 / 70, retos 30 / 30; los retos suman 70 y el tope es 30). Es una estimación mía; la calificación la asigna el profesor.

## Cómo ejecutar mis notebooks

1. Abre cada notebook de `notebooks/` en Google Colab, o en local con un entorno que tenga `langchain-nvidia-ai-endpoints`, `langchain-qdrant`, `qdrant-client`, `markitdown` y `gradio`.
2. Crea el secreto `NVIDIA_API_KEY` en Colab (panel Secrets) con acceso para el notebook, o ponlo en un `.env` local (está en el `.gitignore`).
3. Ejecuta las celdas en orden.

> No incluyo mi API key en ningún archivo de este repositorio.

## Modelo de generación usado

Mi cuenta de NVIDIA devuelve 404 («Function not found for account») para `nvidia/nemotron-3-super-120b-a12b`, el modelo indicado en los notebooks, aunque aparece en la lista de modelos. Usé `nvidia/nemotron-3.5-super-vl-120b-a12b`, del mismo tamaño (120B con 12B activos). Los embeddings sí son los indicados, `nvidia/nemotron-3-embed-1b`. Cada notebook deja un comentario con este cambio.

## Decisiones y aprendizajes

DECISIONES
