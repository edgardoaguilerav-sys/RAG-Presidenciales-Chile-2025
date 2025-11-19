
# RAG Programas Presidenciales Chile 2025
Sistema de consulta inteligente en R + Python que permite consultar los programas presidenciales chilenos 2025 mediante una arquitectura RAG (Retrieval-Augmented Generation). El modelo recupera fragmentos relevantes desde los PDFs oficiales, los pasa por FAISS y genera respuestas usando OpenAI.

## 🚀 Descripción general
- Procesa documentos PDF de programas presidenciales.
- Genera embeddings usando OpenAI.
- Construye un índice vectorial FAISS para consultas rápidas.
- Implementa un motor RAG completo.
- Incluye una aplicación Shiny para realizar consultas interactivas.

## 📁 Estructura del proyecto

app.R → Aplicación Shiny
00_setup_env.R → Configuración de entorno (R + Python)
01_build_index_faiss.R → Construcción del índice FAISS
test_rag_cli.R → Pruebas por consola
rag_build_faiss/ → Índice vectorial + metadata
Programas*.pdf → Documentos fuente


## 🧰 Requisitos

### R:
- shiny  
- reticulate  
- arrow  
- dplyr  
- jsonlite  

### Python:

faiss-cpu
numpy
pandas


## ▶️ Ejecución local

1. Clonar el repositorio:
```bash
git clone https://github.com/TU_USUARIO/RAG-Programas-Presidenciales-2025.git
