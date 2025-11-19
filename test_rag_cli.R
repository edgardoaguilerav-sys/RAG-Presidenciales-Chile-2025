# ============================================================
# test_rag_cli.R — RAG local + OpenAI (REST) usando bge-m3
#   - Embeddings locales con sentence-transformers (BAAI/bge-m3)
#   - Usa el entorno Python configurado en 00_setup_env.R
# ============================================================

suppressPackageStartupMessages({
  library(reticulate)
  library(jsonlite)
  library(arrow)
  library(dplyr)
  library(glue)
  library(tibble)
  library(stringr)
  library(httr2)   # REST al API de OpenAI
})

# -------------------------------------------------------------------
# IMPORTANTE:
# NO volvemos a fijar RETICULATE_PYTHON aquí.
# Asumimos que ya corriste 00_setup_env.R en esta sesión,
# que es quien llama use_python(...) con el entorno rag-faiss.
# -------------------------------------------------------------------
# Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
# NO: Sys.setenv(RETICULATE_PYTHON = "...")

# --- Importes Python (solo para embeddings/FAISS) ---
np    <- import("numpy", delay_load = FALSE, convert = TRUE)
faiss <- import("faiss", delay_load = FALSE, convert = TRUE)

# Modelo local (bge-m3 vía sentence-transformers)
LOCAL_EMB_MODEL <- "BAAI/bge-m3"

# --- Rutas de tu build FAISS ---
OUT_DIR   <- "C:/Users/LENOVO/Desktop/RAG Programas Presidenciales/_rag_build_faiss"
META_FP   <- file.path(OUT_DIR, "meta.json")
CHUNKS_FP <- file.path(OUT_DIR, "chunks.arrow")
INDEX_FP  <- file.path(OUT_DIR, "index.faiss")

stopifnot(file.exists(META_FP), file.exists(CHUNKS_FP), file.exists(INDEX_FP))
meta <- jsonlite::read_json(META_FP)
PROGRAMS <- meta$programs
cat("📚 Programas detectados en meta.json:", paste(PROGRAMS, collapse = ", "), "\n")

# ------------------------------------------------------------
# Embeddings locales con sentence-transformers (BAAI/bge-m3)
# ------------------------------------------------------------

# Embed de un solo texto
embed_one <- function(text) {
  py_code <- sprintf("
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer('%s')
_vecs = _model.encode([%s], normalize_embeddings=True)
", LOCAL_EMB_MODEL, jsonlite::toJSON(as.character(text), auto_unbox = TRUE))
  
  reticulate::py_run_string(py_code, convert = FALSE)
  as.numeric(reticulate::py_eval("_vecs[0].tolist()", convert = TRUE))
}

# Embed de muchos textos (para re-ranking local)
embed_many <- function(texts) {
  texts <- as.character(texts)
  py_code <- sprintf("
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer('%s')
_vecs = _model.encode(%s, normalize_embeddings=True)
", LOCAL_EMB_MODEL, jsonlite::toJSON(texts, auto_unbox = TRUE))
  
  reticulate::py_run_string(py_code, convert = FALSE)
  # Lista de vectores numéricos
  reticulate::py_eval("_vecs.tolist()", convert = TRUE)
}

# Self-test opcional
embed_selftest <- function() {
  cat("\n🧠 Self-test sentence-transformers (bge-m3)...\n")
  txts <- c("educación pública", "crecimiento económico", "seguridad ciudadana")
  vlist <- embed_many(txts)
  cat("✅ bge-m3 vía sentence-transformers funciona. Dimensiones:",
      length(vlist), "x", length(vlist[[1]]), "\n")
}

# ------------------------------------------------------------
# OCR smoke (por si alguno de los PDFs está escaneado)
# ------------------------------------------------------------
ocr_smoke <- function(doc_dir = meta$doc_dir, threshold_chars = 1000) {
  if (is.null(doc_dir) || !dir.exists(doc_dir)) return(invisible(tibble()))
  pdfs <- list.files(doc_dir, pattern = "\\.pdf$", full.names = TRUE)
  tibble(
    archivo = basename(pdfs),
    chars_sin_esp = sapply(pdfs, function(f) {
      txt <- tryCatch(pdftools::pdf_text(f), error = function(e) "")
      nchar(gsub("\\s+", "", paste(txt, collapse = "")))
    }),
    ocr_recomendado = chars_sin_esp < threshold_chars
  ) |>
    arrange(desc(chars_sin_esp))
}

cat("\n🔎 Smoke OCR (posibles PDFs escaneados):\n")
print(ocr_smoke())

# ------------------------------------------------------------
# Contador de consultas
# ------------------------------------------------------------
if (!exists(".ask_count", envir = .GlobalEnv)) .ask_count <<- 0L

# ------------------------------------------------------------
# Util: coseno
# ------------------------------------------------------------
cosine_sim <- function(a, b) {
  a <- as.numeric(a); b <- as.matrix(b)
  denom <- sqrt(sum(a*a)) * sqrt(rowSums(b*b))
  drop((b %*% a) / pmax(denom, 1e-12))
}

# ------------------------------------------------------------
# Función principal: ask(candidate, question)
# ------------------------------------------------------------
ask <- function(candidate, question, k = 8L,
                max_tokens = 700L,
                model = "gpt-4o-mini") {
  if (!candidate %in% PROGRAMS) {
    stop(glue("Programa no válido. Opciones: {paste(PROGRAMS, collapse = ', ')}"))
  }
  key <- Sys.getenv("OPENAI_API_KEY", "")
  if (key == "") stop("Debe definir OPENAI_API_KEY")
  
  .ask_count <<- .ask_count + 1L
  
  # 1) embedding pregunta
  cat(glue("\n🧭 Generando embedding para la pregunta sobre {candidate}...\n"))
  q_vec <- embed_one(question)
  q_np  <- np$expand_dims(np$array(q_vec, dtype = "float32"), 0L)
  
  # 2) FAISS + tabla chunks
  index      <- faiss$read_index(INDEX_FP)
  chunks_tbl <- arrow::read_feather(CHUNKS_FP)
  
  # 3) búsqueda amplia y FILTRO ESTRICTO por candidato
  k_search <- min(nrow(chunks_tbl), max(256L, k * 16L))
  res  <- index$search(q_np, as.integer(k_search))
  idx  <- as.integer(res[[2]][1, ])
  topk_global <- chunks_tbl[idx + 1, ] |>
    dplyr::select(program, title, page_display, chunk)
  
  topk_prog <- dplyr::filter(topk_global, .data$program == candidate)
  
  # 4) plan B: re-rank local SOLO dentro del candidato
  if (nrow(topk_prog) < k) {
    prog_all <- dplyr::filter(chunks_tbl, .data$program == candidate)
    cat(glue("🧪 Re-rankeando dentro de {candidate} con bge-m3 (local)...\n"))
    embs <- embed_many(prog_all$chunk)
    M    <- do.call(rbind, embs)
    sims <- cosine_sim(q_vec, M)
    ord  <- order(sims, decreasing = TRUE)
    topk_prog <- prog_all[ord, , drop = FALSE]
  }
  topk_prog <- topk_prog[seq_len(min(k, nrow(topk_prog))), , drop = FALSE]
  
  topk_prog <- dplyr::filter(topk_prog, .data$program == candidate)
  
  if (nrow(topk_prog) == 0) {
    msg <- glue("no se menciona en el programa de gobierno de {candidate}")
    cat("\n⚠️ Aviso:", msg, "\n"); return(invisible(msg))
  }
  
  cat("\n🔎 Fragmentos más relevantes (", candidate, "):\n", sep = "")
  print(topk_prog)
  
  # 5) contexto con página visible
  pg <- ifelse(is.na(topk_prog$page_display) | topk_prog$page_display == "",
               "p.?", topk_prog$page_display)
  ctx_lines    <- sprintf("[[%s]] %s", pg, topk_prog$chunk)
  context_text <- paste(ctx_lines, collapse = "\n---\n")
  
  instruccion_formato <- glue(
    "Responde en español con viñetas numeradas. DEBES entregar respuestas completas, no entregues ideas a medio desarrollar. ",
    "Cada viñeta DEBE terminar con la cita (Pág. XX; Programa de {candidate}). ",
    "No inventes páginas ni uses info fuera de los fragmentos. Las respuestas DEBEN relacionarse exclusivamente con la temática de la pregunta."
  )
  user_prompt <- paste0(
    instruccion_formato, "\n\n",
    "Pregunta del usuario: ", question, "\n\n",
    "Fragmentos del programa de ", candidate, " (cada línea lleva su página entre [[ ]]):\n",
    context_text
  )
  
  # 6) limpiar proxies para evitar errores de red
  for (v in c("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy",
              "ALL_PROXY","all_proxy","OPENAI_PROXY","openai_proxy")) {
    if (nzchar(Sys.getenv(v, ""))) Sys.unsetenv(v)
  }
  
  # 7) llamada a OpenAI por REST
  req <- httr2::request("https://api.openai.com/v1/chat/completions") |>
    httr2::req_headers(
      Authorization = paste("Bearer", key),
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(list(
      model = model,
      messages = list(
        list(role = "system",
             content = "Eres un analista que cita con precisión los fragmentos proveídos."),
        list(role = "user", content = user_prompt)
      ),
      max_tokens = as.integer(max_tokens)
    ))
  resp <- httr2::req_perform(req)
  j    <- httr2::resp_body_json(resp)
  
  if (is.null(j$choices) || length(j$choices) == 0) {
    msg <- glue("no se menciona en el programa de gobierno de {candidate}")
    cat("\n⚠️ Aviso:", msg, "\n"); return(invisible(msg))
  }
  ans <- j$choices[[1]]$message$content %||% ""
  cat(glue("\n💬 Consulta número {.ask_count} — respuesta generada:\n"))
  cat(ans, "\n")
  invisible(ans)
}

# ------------------------------------------------------------
# Ejemplo rápido (recuerda no dejar la API key fija en el script)
# ------------------------------------------------------------
ask("Kast", "¿Qué derechos humanos se ven comprometidos con su programa?")
