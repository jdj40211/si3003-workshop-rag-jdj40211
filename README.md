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
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ✅ | [notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb), [RESPUESTAS.md](RESPUESTAS.md) |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ✅ | [notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb), [RESPUESTAS.md](RESPUESTAS.md) |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | [notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb), [RESPUESTAS.md](RESPUESTAS.md) |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | [notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb), [RESPUESTAS.md](RESPUESTAS.md) |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | [notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb), [RESPUESTAS.md](RESPUESTAS.md) |
| R1 | Reflexión final con tus datos | 10 | ✅ | [RESPUESTAS.md](RESPUESTAS.md) |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | [gráficas](evidencias/C1_recall_efsearch.png), [04_challenges](notebooks/04_challenges.ipynb) |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ✅ | [04_challenges](notebooks/04_challenges.ipynb) |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ✅ | [C3_gradio.png](evidencias/C3_gradio.png), [04_challenges](notebooks/04_challenges.ipynb) |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ✅ | [eval/preguntas.csv](eval/preguntas.csv), [04_challenges](notebooks/04_challenges.ipynb) |
| C5 | MMR frente a similitud pura (reto) | 8 | ✅ | [04_challenges](notebooks/04_challenges.ipynb) |
| C6 | Persistencia del índice en disco (reto) | 8 | ✅ | [04_challenges](notebooks/04_challenges.ipynb) |

**Autoevaluación:** 100 / 100 puntos (núcleo 70 / 70, retos 30 / 30; los retos suman 70 y el tope es 30). Es una estimación mía; la calificación la asigna el profesor.

## Cómo ejecutar mis notebooks

1. Abre cada notebook de `notebooks/` en Google Colab, o en local con un entorno que tenga `langchain-nvidia-ai-endpoints`, `langchain-qdrant`, `qdrant-client`, `markitdown` y `gradio`.
2. Crea el secreto `NVIDIA_API_KEY` en Colab (panel Secrets) con acceso para el notebook, o ponlo en un `.env` local (está en el `.gitignore`).
3. Ejecuta las celdas en orden.

> No incluyo mi API key en ningún archivo de este repositorio.

## Modelo de generación usado

Mi cuenta de NVIDIA devuelve 404 («Function not found for account») para `nvidia/nemotron-3-super-120b-a12b`, el modelo indicado en los notebooks, aunque aparece en la lista de modelos. Usé `nvidia/nemotron-3.5-super-vl-120b-a12b`, del mismo tamaño (120B con 12B activos). Los embeddings sí son los indicados, `nvidia/nemotron-3-embed-1b`. Cada notebook deja un comentario con este cambio.

## Decisiones y aprendizajes

- **El umbral del notebook no filtraba nada.** `score_threshold=0.3` se compara en Qdrant contra `(coseno + 1) / 2`, no contra el coseno. Por eso las preguntas fuera de contexto devolvían siempre 3 fuentes. Con mis datos propuse coseno >= 0.30, es decir `score_threshold=0.65`, que conserva las 8 preguntas buenas y deja sin contexto a 3 de las 4 que no se podían responder. No existe un umbral perfecto: la peor pregunta buena sacó 0.354 y la mejor fuera de contexto 0.393.
- **El tamaño del chunk depende del corpus.** Mis 11 documentos de NodoIA son cortos. Con `chunk_size=400` salían 18 chunks y 7 documentos partidos; con 700 sale un chunk por documento.
- **Del prompt, lo que más cambió fue pedir la cita entre corchetes.** El límite de 3 oraciones fue lo que el modelo siguió peor, y la frase fija se aplicó a veces de más, incluso con contexto relevante.
- **Evaluar a ojo no escala.** Con una métrica de recuperación (hit@k) sobre 10 preguntas, `chunk_size=900` dio hit@3 de 90 %, contra 70 % con 400 y con 1500. La métrica tiene límites: busca una palabra clave, así que castiga respuestas bien recuperadas con otra redacción.
- **HNSW no gana siempre.** Con vectores aleatorios necesité `efSearch=256` para un recall de 0.97 con M=16, y a recall igual no le ganó al índice exacto hasta n=500.000. Con datos agrupados, más parecidos a embeddings reales, `efSearch=10` ya daba 0.916. El tipo de datos importa tanto como los parámetros.
- **MMR y persistencia.** MMR bajó la redundancia en las 3 preguntas probadas, pero con 10 preguntas el hit@5 quedó igual. Reabrir un índice de Qdrant en disco tardó unos 19 ms, contra unos 8,7 s de volver a indexar.
- **Mi cuenta de NVIDIA no tenía habilitado el modelo del curso.** El 404 apareció solo al invocarlo, no al listar los modelos, así que tuve que probar los modelos uno a uno y quedarme con uno del mismo tamaño. Aprendí a comprobar el acceso a cada modelo antes de depender de él.
