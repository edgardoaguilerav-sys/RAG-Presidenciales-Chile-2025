# RAG Programas Presidenciales (Chile 2025)
Consultas a programas presidenciales usando un pipeline **RAG local**:
- **Embeddings + FAISS** para búsqueda amplia (recall alto)
- **Cross-Encoder** para re-ranking (precisión alta)
- Respuesta con **citas obligatorias por página**
- Generación con **OpenAI** (si hay API key) y fallback a **Ollama** (local)
- Parámetro **temperature** controlado (por defecto 0.1) para respuestas más estables

> Objetivo: que las respuestas estén **ancladas a fragmentos del programa seleccionado**, evitando mezclar candidatos y reduciendo “alucinaciones” del modelo.

---

## Demo (Shiny)
La app permite:
1) elegir candidato  
2) escribir una pregunta  
3) recibir una lista numerada con **(Programa X; Pág. N)** al final de cada ítem

**Características UI (para capturas):**
- sin scroll horizontal en la respuesta
- texto más oscuro y legible
- bloqueo de inputs mientras genera
- mensaje “Generando respuesta usando el modelo X…”

---

## Arquitectura (alto nivel)

1. **Ingesta / chunking** (script de build)
   - Se generan fragmentos (chunks) desde PDFs de programas.
2. **Embeddings**
   - Se calcula embedding para cada chunk.
3. **Indexación FAISS**
   - Se construye un índice vectorial global.
4. **Consulta**
   - Se hace búsqueda ancha en FAISS (global).
   - Se **filtra por candidato** (fail-fast anti mezcla).
   - Se aplica **Cross-Encoder** para ordenar los chunks relevantes.
   - Se construye un contexto con marcadores `[[p.X]]`.
5. **Generación**
   - OpenAI si `OPENAI_API_KEY` existe y responde; si no, Ollama.
   - Prompt restringido a **lista numerada**, sin markdown.
6. **Postproceso**
   - Si el modelo no cita correctamente, se **fuerzan citas por código** mapeando cada ítem al chunk más relevante con Cross-Encoder.

---

## Requisitos
- Windows (probado en Windows 10/11)
- **R + RStudio**
- **Miniconda / conda**
- Paquetes R: `reticulate`, `arrow`, `dplyr`, `stringr`, `httr2`, `jsonlite`, `shiny`, `shinythemes`, `shinyjs`
- Python env (conda): `faiss-cpu`, `sentence-transformers`, `numpy`

Opcional:
- Cuenta OpenAI (para usar la API)
- **Ollama** instalado y corriendo (fallback local)

---

## Instalación rápida

### 1) Crear/activar el entorno conda
Este repo asume un env llamado:

- `rag-faiss`

Puedes usar tu script `00_setup_env.R` (recomendado) o crearlo manualmente.

**Manual (referencial):**
```bash
conda create -n rag-faiss python=3.11 -y
conda activate rag-faiss
pip install -U numpy faiss-cpu sentence-transformers
