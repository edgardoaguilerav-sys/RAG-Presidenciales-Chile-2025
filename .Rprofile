local({
  # Fijar siempre el Python del entorno rag-faiss
  Sys.setenv(
    RETICULATE_AUTOCONFIGURE = "FALSE",
    RETICULATE_PYTHON =
      "C:/Users/LENOVO/AppData/Local/r-miniconda/envs/rag-faiss/python.exe"
  )
  
  message("⚙️ RETICULATE_PYTHON fijado a entorno 'rag-faiss'.")
  
  # Cargar el script RAG (define PROGRAMS y ask())
  source("test_rag_cli.R", local = FALSE)
  message("✅ RAG listo: usa ask(candidato, pregunta) o ejecuta app.R")
})
