# ============================================================
# test_rag_cli.R — RAG local + FAISS + generación (OpenAI -> fallback Ollama)
#   - Embeddings: mismo modelo usado en el índice FAISS (desde meta.json)
#   - Recuperación: FAISS
#   - Respuesta:
#        1) OpenAI si OPENAI_API_KEY existe y la llamada funciona
#        2) Si no, Ollama local (si está instalado y el server responde)
#   - Citas: se fuerzan por CÓDIGO si el LLM no las pone (robusto para Ollama)
#   - Logging: imprime "Modelo usado: <provider> — <model>"
# ============================================================

suppressPackageStartupMessages({
  library(reticulate)
  library(jsonlite)
  library(arrow)
  library(dplyr)
  library(glue)
  library(stringr)
  library(httr2)
})

`%||%` <- function(x, y) if (!is.null(x)) x else y

# ============================================================
# 0) Python/conda: detectar conda y apuntar al env rag-faiss
# (si reticulate ya inicializó otro Python en esta sesión: Session -> Restart R)
# ============================================================
env_name <- "rag-faiss"

cb <- tryCatch(reticulate::conda_binary(), error = function(e) NULL)
if (is.null(cb)) stop("No se detectó conda. Ejecuta 00_setup_env.R primero.")

miniconda_dir <- normalizePath(dirname(dirname(cb)), winslash = "/")
py_exe <- file.path(miniconda_dir, "envs", env_name, "python.exe")
if (!file.exists(py_exe)) stop("No existe python.exe del env '", env_name, "': ", py_exe)

Sys.setenv(RETICULATE_MINICONDA_PATH = miniconda_dir)
Sys.setenv(RETICULATE_PYTHON = py_exe)

use_python(py_exe, required = TRUE)
py_config()

# ============================================================
# 1) Rutas índice
# ============================================================
OUT_DIR   <- "C:/Users/LENOVO/Desktop/RAG Programas Presidenciales/_rag_build_faiss"
META_FP   <- file.path(OUT_DIR, "meta.json")
CHUNKS_FP <- file.path(OUT_DIR, "chunks.arrow")
INDEX_FP  <- file.path(OUT_DIR, "index.faiss")

stopifnot(file.exists(META_FP), file.exists(CHUNKS_FP), file.exists(INDEX_FP))

meta <- jsonlite::read_json(META_FP, simplifyVector = TRUE)

PROGRAMS <- meta$programs
cat("📚 Programas detectados:", paste(PROGRAMS, collapse = ", "), "\n")

LOCAL_EMB_MODEL <- meta$emb$local_model
if (is.null(LOCAL_EMB_MODEL) || !nzchar(LOCAL_EMB_MODEL)) {
  stop("meta.json no trae emb.local_model. Usa el índice generado por tu 01 actualizado.")
}
cat("🧠 Embedding model (meta.json):", LOCAL_EMB_MODEL, "\n")

# ============================================================
# 2) Python: FAISS + SentenceTransformers
# ============================================================
np    <- import("numpy", convert = TRUE)
faiss <- import("faiss", convert = TRUE)
st    <- import("sentence_transformers", convert = TRUE)

embedder <- st$SentenceTransformer(LOCAL_EMB_MODEL, device = "cpu")

# ---- Embeddings robustos (evitan v[1, ] cuando viene 1D) ----
embed_one <- function(text) {
  v <- embedder$encode(
    as.character(text),
    show_progress_bar = FALSE,
    convert_to_numpy = TRUE,
    normalize_embeddings = TRUE
  )
  dv <- dim(v)
  if (is.null(dv) || length(dv) < 2) return(as.numeric(v))
  as.numeric(v[1, , drop = TRUE])
}

embed_many <- function(texts, batch_size = 64L) {
  texts <- as.character(texts)
  v <- embedder$encode(
    texts,
    batch_size = as.integer(batch_size),
    show_progress_bar = FALSE,
    convert_to_numpy = TRUE,
    normalize_embeddings = TRUE
  )
  dv <- dim(v)
  if (is.null(dv) || length(dv) < 2) return(matrix(as.numeric(v), nrow = 1))
  as.matrix(v)
}

# Util coseno
cosine_sim <- function(a, b) {
  a <- as.numeric(a); b <- as.matrix(b)
  denom <- sqrt(sum(a*a)) * sqrt(rowSums(b*b))
  drop((b %*% a) / pmax(denom, 1e-12))
}

# ============================================================
# 3) Backends de generación: OpenAI + Ollama
#    - Devuelven el texto con atributos: used_provider / used_model
# ============================================================
openai_key_available <- function() nzchar(Sys.getenv("OPENAI_API_KEY", ""))

generate_openai <- function(prompt, model = "gpt-4o-mini", max_tokens = 700L) {
  key <- Sys.getenv("OPENAI_API_KEY", "")
  if (!nzchar(key)) stop("OPENAI_API_KEY no definida.")
  
  req <- httr2::request("https://api.openai.com/v1/chat/completions") |>
    httr2::req_headers(
      Authorization = paste("Bearer", key),
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(list(
      model = model,
      messages = list(
        list(role = "system", content = "Responde solo con evidencia de los fragmentos. No inventes."),
        list(role = "user", content = prompt)
      ),
      max_tokens = as.integer(max_tokens)
    )) |>
    httr2::req_timeout(60)
  
  resp <- httr2::req_perform(req)
  j <- httr2::resp_body_json(resp)
  txt <- j$choices[[1]]$message$content
  
  attr(txt, "used_provider") <- "OpenAI"
  attr(txt, "used_model") <- model
  txt
}

ollama_installed <- function() {
  tryCatch({
    status <- suppressWarnings(system("ollama --version", ignore.stdout = TRUE, ignore.stderr = TRUE))
    isTRUE(status == 0)
  }, error = function(e) FALSE)
}

# MUY importante en Windows: 127.0.0.1 evita problemas con "localhost"
OLLAMA_BASE_URL <- Sys.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

ollama_server_up <- function() {
  tryCatch({
    httr2::request(paste0(OLLAMA_BASE_URL, "/api/tags")) |>
      httr2::req_timeout(3) |>
      httr2::req_perform()
    TRUE
  }, error = function(e) FALSE)
}

generate_ollama <- function(prompt, model = "llama3.2:3b",
                            timeout_sec = 180,
                            options = list(temperature = 0.1, top_p = 0.9, num_predict = 700)) {
  if (!ollama_installed()) stop("Ollama no está instalado (comando 'ollama' no encontrado).")
  if (!ollama_server_up()) stop("Ollama está instalado, pero el servidor no responde en ", OLLAMA_BASE_URL)
  
  body <- list(
    model = model,
    prompt = prompt,
    stream = FALSE,
    options = options
  )
  
  resp <- httr2::request(paste0(OLLAMA_BASE_URL, "/api/generate")) |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(timeout_sec) |>
    httr2::req_perform()
  
  j <- httr2::resp_body_json(resp)
  txt <- j$response
  
  attr(txt, "used_provider") <- "Ollama"
  attr(txt, "used_model") <- model
  txt
}

# AUTO: intenta OpenAI si hay key; si falla o no hay key -> Ollama
generate_answer <- function(prompt,
                            openai_model = "gpt-4o-mini",
                            openai_max_tokens = 700L,
                            ollama_model = "llama3.2:3b") {
  if (openai_key_available()) {
    out <- tryCatch(
      generate_openai(prompt, model = openai_model, max_tokens = openai_max_tokens),
      error = function(e) {
        message("⚠️ OpenAI falló: ", conditionMessage(e))
        NULL
      }
    )
    if (!is.null(out)) return(out)
    message("➡️ Probando Ollama como fallback...")
  } else {
    message("ℹ️ No hay OPENAI_API_KEY. Usando Ollama...")
  }
  
  generate_ollama(prompt, model = ollama_model)
}

# ============================================================
# 3.5) Post-proceso: forzar citas por código (robusto para Ollama)
# ============================================================

extract_page_num <- function(page_display) {
  if (is.na(page_display) || !nzchar(page_display)) return(NA_integer_)
  x <- str_extract(as.character(page_display), "\\d+")
  if (is.na(x)) return(NA_integer_)
  as.integer(x)
}

has_program_cite <- function(x) {
  grepl("\\(\\s*Programa\\s+.+?;\\s*P[aá]g", x, ignore.case = TRUE)
}

# Detecta inicio de viñeta: "1. ..." o "- ..." o "• ..."
is_bullet_start <- function(line) {
  grepl("^\\s*(\\d+\\.|[-*•])\\s+", line)
}

strip_bullet_prefix <- function(line) {
  sub("^\\s*(\\d+\\.|[-*•])\\s+", "", line)
}

fmt_cite <- function(candidate, pages) {
  pages <- pages[!is.na(pages)]
  pages <- unique(pages)
  if (!length(pages)) return(glue("(Programa {candidate}; Pág. ?)"))
  pages <- sort(pages)
  if (length(pages) == 1) glue("(Programa {candidate}; Pág. {pages[[1]]})")
  else glue("(Programa {candidate}; Págs. {paste(pages, collapse = ', ')})")
}

# Agrega citas a cada viñeta si faltan (elige páginas por similitud con chunks recuperados)
ensure_citations <- function(ans, candidate, topk_prog) {
  if (is.null(ans) || !nzchar(ans) || nrow(topk_prog) == 0) return(ans)
  
  lines <- strsplit(ans, "\n", fixed = TRUE)[[1]]
  if (!any(vapply(lines, is_bullet_start, logical(1)))) return(ans)
  
  # Pre-embeddings de chunks para comparar rápido
  chunk_mat <- embed_many(topk_prog$chunk, batch_size = 64L)
  page_nums <- vapply(topk_prog$page_display, extract_page_num, integer(1))
  
  out_lines <- character(0)
  i <- 1L
  while (i <= length(lines)) {
    if (!is_bullet_start(lines[i])) {
      out_lines <- c(out_lines, lines[i])
      i <- i + 1L
      next
    }
    
    # Captura bloque de viñeta (puede ocupar varias líneas)
    j <- i + 1L
    while (j <= length(lines) && !is_bullet_start(lines[j])) j <- j + 1L
    bullet_block <- lines[i:(j - 1L)]
    
    bullet_text <- paste(vapply(bullet_block, strip_bullet_prefix, character(1)), collapse = " ")
    bullet_text <- str_squish(bullet_text)
    
    # Si ya tiene cita, lo dejamos tal cual
    if (has_program_cite(paste(bullet_block, collapse = "\n"))) {
      out_lines <- c(out_lines, bullet_block)
      i <- j
      next
    }
    
    # Elegir páginas más probables por similitud
    bvec <- embed_one(bullet_text)
    sims <- cosine_sim(bvec, chunk_mat)
    ord <- order(sims, decreasing = TRUE)
    
    # 1 ó 2 páginas si están muy cercanas
    best_idx <- ord[1]
    pages <- page_nums[best_idx]
    if (length(ord) >= 2) {
      second_idx <- ord[2]
      if (!is.na(sims[second_idx]) && (sims[best_idx] - sims[second_idx]) < 0.02) {
        pages <- c(pages, page_nums[second_idx])
      }
    }
    
    cite <- fmt_cite(candidate, pages)
    
    # Agrega la cita al final de la ÚLTIMA línea del bloque
    bullet_block[length(bullet_block)] <- paste0(rtrim(bullet_block[length(bullet_block)]), " ", cite)
    out_lines <- c(out_lines, bullet_block)
    
    i <- j
  }
  
  paste(out_lines, collapse = "\n")
}

rtrim <- function(x) sub("\\s+$", "", x)

# ============================================================
# 4) ASK principal
# ============================================================
ask <- function(candidate, question,
                k = 8L,
                openai_model = "gpt-4o-mini",
                openai_max_tokens = 700L,
                ollama_model = "llama3.2:3b",
                print_console = TRUE) {
  
  if (!candidate %in% PROGRAMS) {
    stop(glue("Programa no válido. Opciones: {paste(PROGRAMS, collapse=', ')}"))
  }
  
  # 1) embedding de la pregunta (misma base que el índice)
  q_vec <- embed_one(question)
  q_np  <- np$expand_dims(np$array(q_vec, dtype = "float32"), 0L)
  
  # 2) cargar FAISS + chunks
  index <- faiss$read_index(INDEX_FP)
  chunks_tbl <- arrow::read_feather(CHUNKS_FP)
  
  # 3) búsqueda amplia
  k_search <- min(nrow(chunks_tbl), max(256L, k * 16L))
  res <- index$search(q_np, as.integer(k_search))
  idx <- as.integer(res[[2]][1, ])
  idx <- idx[idx >= 0]
  
  if (!length(idx)) {
    msg <- "No se encontró evidencia en el índice para la pregunta (global)."
    message(msg)
    return(invisible(msg))
  }
  
  topk_global <- chunks_tbl[idx + 1, ] |>
    dplyr::select(program, title, page_display, chunk)
  
  topk_prog <- dplyr::filter(topk_global, program == candidate)
  
  # fallback rerank si hay pocos
  if (nrow(topk_prog) < k) {
    prog_all <- dplyr::filter(chunks_tbl, program == candidate)
    if (nrow(prog_all) == 0) {
      msg <- glue("No hay chunks para el programa de {candidate}.")
      message(msg)
      return(invisible(msg))
    }
    
    M <- embed_many(prog_all$chunk, batch_size = 64L)
    sims <- cosine_sim(q_vec, M)
    ord <- order(sims, decreasing = TRUE)
    topk_prog <- prog_all[ord, , drop = FALSE]
  }
  
  topk_prog <- topk_prog[seq_len(min(k, nrow(topk_prog))), ]
  if (nrow(topk_prog) == 0) {
    msg <- glue("No se menciona este tema en el programa de {candidate}.")
    message(msg)
    return(invisible(msg))
  }
  
  # Contexto con páginas (manteniendo [[p.X]] para que el modelo elija X)
  pg_marker <- ifelse(is.na(topk_prog$page_display) | topk_prog$page_display == "",
                      "p.?", topk_prog$page_display)
  ctx_lines <- sprintf("[[%s]] %s", pg_marker, topk_prog$chunk)
  context_text <- paste(ctx_lines, collapse = "\n---\n")
  
  # ============================================================
  # PROMPT (AQUÍ empieza y termina el prompt)
  # ============================================================
  prompt <- glue("
Responde en español con una lista numerada usando el formato exacto:
1. ...
2. ...
etc.

Formato OBLIGATORIO:
- Cada viñeta debe terminar EXACTAMENTE con: (Programa {candidate}; Pág. X)
- Donde X es el número de página tomado del marcador [[p.X]] de los fragmentos usados.
- Si usas más de una página en la misma viñeta, usa: (Programa {candidate}; Págs. X, Y)

Reglas:
- Usa SOLO información contenida en los fragmentos.
- No inventes ni completes con conocimiento externo.
- No incluyas marcadores [[p.X]] en el texto final: solo el formato (Programa ...; Pág. X).

Pregunta del usuario: {question}

Fragmentos recuperados (Programa de {candidate}):
{context_text}
")
  # ============================================================
  # FIN PROMPT
  # ============================================================
  
  ans <- generate_answer(
    prompt = prompt,
    openai_model = openai_model,
    openai_max_tokens = openai_max_tokens,
    ollama_model = ollama_model
  )
  
  used_provider <- attr(ans, "used_provider") %||% "Desconocido"
  used_model    <- attr(ans, "used_model") %||% "Desconocido"
  
  # Fuerza citas por código (clave para Shiny + Ollama)
  ans2 <- ensure_citations(ans, candidate, topk_prog)
  
  # Mantener el metadata del backend también en el resultado final
  attr(ans2, "used_provider") <- used_provider
  attr(ans2, "used_model")    <- used_model
  
  if (isTRUE(print_console)) {
    cat("\n=====================\n")
    cat("Modelo usado: ", used_provider, " — ", used_model, "\n", sep = "")
    cat("Respuesta generada:\n")
    cat(ans2)
    cat("\n=====================\n")
  }
  
  invisible(ans2)
}

ask("Parisi", "¿Qué medidas plantea en temas de seguridad e inmigración ilegal?")
