# ============================================================
# test_rag_cli.R — RAG local + FAISS + generación (OpenAI -> fallback Ollama)
#   - Embeddings: mismo modelo usado en el índice FAISS (desde meta.json)
#   - Recuperación: FAISS (ancha)
#   - Re-rank: Cross-Encoder (reemplaza coseno preciso)
#   - Respuesta:
#        1) OpenAI si OPENAI_API_KEY existe y la llamada funciona
#        2) Si no, Ollama local (si está instalado y el server responde)
#   - Citas: se fuerzan por CÓDIGO si el LLM no las pone (robusto para Ollama)
#            + permite cita “Pág. N” o “Págs. N, N+1” para bordes de página
#   - Logging: imprime "Modelo usado: <provider> — <model>"
#   - Temperature: parámetro en ask() (default 0.1)
#
#   PATCH MÍNIMO (anti “salidas raras” + timeouts):
#     - OpenAI timeout subido a 120s
#     - Prompt: prohíbe markdown/negritas/títulos e introducciones; SOLO lista numerada
#     - ensure_citations(): detecta SOLO viñetas numeradas (evita que '*' corte la salida)
#     - Normalización + fail-fast: candidate se canoniza y se valida que el contexto sea SOLO del candidato
#
#   PATCH MÍNIMO (anti duplicados):
#     - Prompt: prohíbe repetir ideas; si solo hay 1 punto sustentado -> devolver 1 ítem
#     - Post: dedupe_numbered_list() elimina ítems numerados duplicados y renumera
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
# 1.5) Normalización de candidato (evita mezclas por nombres)
# ============================================================
norm_key <- function(x) tolower(str_squish(as.character(x)))

canonical_program <- function(candidate) {
  cand_key <- norm_key(candidate)
  prog_keys <- norm_key(PROGRAMS)
  hit <- which(prog_keys == cand_key)
  if (length(hit) != 1) {
    stop(glue("Programa no válido o ambiguo: '{candidate}'. Opciones: {paste(PROGRAMS, collapse=', ')}"))
  }
  PROGRAMS[[hit]]
}

# ============================================================
# 2) Python: FAISS + SentenceTransformers (+ CrossEncoder)
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

# Util coseno (se mantiene solo para “tiebreaks” internos si hiciera falta)
cosine_sim <- function(a, b) {
  a <- as.numeric(a); b <- as.matrix(b)
  denom <- sqrt(sum(a*a)) * sqrt(rowSums(b*b))
  drop((b %*% a) / pmax(denom, 1e-12))
}

# ============================================================
# 2.5) Cross-Encoder (re-rank) — modelo por defecto configurable
# ============================================================
CROSSENCODER_MODEL <- Sys.getenv(
  "RAG_CROSSENCODER_MODEL",
  "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

cross_encoder <- NULL
get_cross_encoder <- function() {
  if (!is.null(cross_encoder)) return(cross_encoder)
  ce <- st$CrossEncoder(CROSSENCODER_MODEL, device = "cpu")
  assign("cross_encoder", ce, envir = .GlobalEnv)
  ce
}

cross_rerank <- function(query, docs, batch_size = 32L) {
  ce <- get_cross_encoder()
  pairs <- lapply(as.character(docs), function(d) list(as.character(query), as.character(d)))
  as.numeric(ce$predict(pairs, batch_size = as.integer(batch_size), show_progress_bar = FALSE))
}

# ============================================================
# 3) Backends de generación: OpenAI + Ollama
#    - Devuelven texto con atributos: used_provider / used_model
#    - Temperature parametrizable
# ============================================================
openai_key_available <- function() nzchar(Sys.getenv("OPENAI_API_KEY", ""))

generate_openai <- function(prompt, model = "gpt-4o-mini", max_tokens = 700L, temperature = 0.1) {
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
      temperature = as.numeric(temperature),
      max_tokens = as.integer(max_tokens)
    )) |>
    httr2::req_timeout(120)  # PATCH: antes 60
  
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
                            temperature = 0.1,
                            options = list(top_p = 0.9, num_predict = 700)) {
  if (!ollama_installed()) stop("Ollama no está instalado (comando 'ollama' no encontrado).")
  if (!ollama_server_up()) stop("Ollama está instalado, pero el servidor no responde en ", OLLAMA_BASE_URL)
  
  options2 <- options
  options2$temperature <- as.numeric(temperature)
  
  body <- list(
    model = model,
    prompt = prompt,
    stream = FALSE,
    options = options2
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

generate_answer <- function(prompt,
                            openai_model = "gpt-4o-mini",
                            openai_max_tokens = 700L,
                            ollama_model = "llama3.2:3b",
                            temperature = 0.1) {
  if (openai_key_available()) {
    out <- tryCatch(
      generate_openai(prompt, model = openai_model, max_tokens = openai_max_tokens, temperature = temperature),
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
  
  generate_ollama(prompt, model = ollama_model, temperature = temperature)
}

# ============================================================
# 3.5) Post-proceso: forzar citas por código
#   - usa Cross-Encoder para mapear viñeta -> chunk(s) más relevantes
#   - bordes de página: N o N,N+1
#   - PATCH: detecta SOLO viñetas numeradas (evita '*' como bullet)
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

# PATCH: SOLO numeradas "1. ..."
is_bullet_start <- function(line) {
  grepl("^\\s*\\d+\\.\\s+", line)
}

strip_bullet_prefix <- function(line) {
  sub("^\\s*\\d+\\.\\s+", "", line)
}

fmt_cite <- function(candidate, pages) {
  pages <- pages[!is.na(pages)]
  pages <- unique(pages)
  if (!length(pages)) return(glue("(Programa {candidate}; Pág. ?)"))
  pages <- sort(pages)
  if (length(pages) == 1) glue("(Programa {candidate}; Pág. {pages[[1]]})")
  else glue("(Programa {candidate}; Págs. {paste(pages, collapse = ', ')})")
}

rtrim <- function(x) sub("\\s+$", "", x)

near_page_boundary <- function(p1, p2, s1, s2, score_delta = 0.15) {
  if (is.na(p1) || is.na(p2)) return(FALSE)
  if (abs(p1 - p2) != 1) return(FALSE)
  if (is.na(s1) || is.na(s2)) return(FALSE)
  (s1 - s2) <= score_delta
}

ensure_citations <- function(ans, candidate, topk_prog, question_for_ce = NULL) {
  if (is.null(ans) || !nzchar(ans) || nrow(topk_prog) == 0) return(ans)
  
  lines <- strsplit(ans, "\n", fixed = TRUE)[[1]]
  if (!any(vapply(lines, is_bullet_start, logical(1)))) return(ans)
  
  page_nums <- vapply(topk_prog$page_display, extract_page_num, integer(1))
  chunks_txt <- as.character(topk_prog$chunk)
  
  out_lines <- character(0)
  i <- 1L
  while (i <= length(lines)) {
    if (!is_bullet_start(lines[i])) {
      out_lines <- c(out_lines, lines[i])
      i <- i + 1L
      next
    }
    
    j <- i + 1L
    while (j <= length(lines) && !is_bullet_start(lines[j])) j <- j + 1L
    bullet_block <- lines[i:(j - 1L)]
    
    bullet_text <- paste(vapply(bullet_block, strip_bullet_prefix, character(1)), collapse = " ")
    bullet_text <- str_squish(bullet_text)
    
    if (has_program_cite(paste(bullet_block, collapse = "\n"))) {
      out_lines <- c(out_lines, bullet_block)
      i <- j
      next
    }
    
    query_ce <- (question_for_ce %||% bullet_text)
    scores <- cross_rerank(query_ce, chunks_txt, batch_size = 32L)
    ord <- order(scores, decreasing = TRUE)
    
    best_idx <- ord[1]
    p1 <- page_nums[best_idx]
    s1 <- scores[best_idx]
    
    pages <- p1
    
    if (length(ord) >= 2) {
      second_idx <- ord[2]
      p2 <- page_nums[second_idx]
      s2 <- scores[second_idx]
      if (near_page_boundary(p1, p2, s1, s2, score_delta = 0.15)) {
        pages <- c(p1, p2)
      }
    }
    
    cite <- fmt_cite(candidate, pages)
    bullet_block[length(bullet_block)] <- paste0(rtrim(bullet_block[length(bullet_block)]), " ", cite)
    out_lines <- c(out_lines, bullet_block)
    
    i <- j
  }
  
  paste(out_lines, collapse = "\n")
}

# ============================================================
# 3.6) PATCH anti-duplicados: elimina ítems numerados duplicados
#      (sin inventar contenido) y renumera
# ============================================================
dedupe_numbered_list <- function(ans) {
  if (is.null(ans) || !nzchar(ans)) return(ans)
  
  lines <- strsplit(ans, "\n", fixed = TRUE)[[1]]
  is_start <- function(x) grepl("^\\s*\\d+\\.\\s+", x)
  if (!any(vapply(lines, is_start, logical(1)))) return(ans)
  
  # Construir salida preservando líneas no numeradas previas (por si aparecen)
  prefix_lines <- character(0)
  idx_first <- which(vapply(lines, is_start, logical(1)))[1]
  if (idx_first > 1) prefix_lines <- lines[1:(idx_first - 1)]
  
  # Partir en bloques por ítem numerado desde idx_first
  blocks <- list()
  i <- idx_first
  while (i <= length(lines)) {
    if (!is_start(lines[i])) { i <- i + 1L; next }
    j <- i + 1L
    while (j <= length(lines) && !is_start(lines[j])) j <- j + 1L
    blocks[[length(blocks) + 1L]] <- lines[i:(j - 1L)]
    i <- j
  }
  
  # Key normalizada (sin cita / sin tildes / sin signos) para dedupe
  key_of <- function(block) {
    txt <- paste(block, collapse = " ")
    txt <- gsub("\\(\\s*Programa\\s+.+?;\\s*P[aá]gs?\\..+?\\)", "", txt, ignore.case = TRUE) # quita cita
    txt <- tolower(trimws(gsub("\\s+", " ", txt)))
    txt <- iconv(txt, to = "ASCII//TRANSLIT")
    txt <- gsub("[^a-z0-9 ]+", " ", txt)
    txt <- trimws(gsub("\\s+", " ", txt))
    txt
  }
  
  keys <- vapply(blocks, key_of, character(1))
  keep_idx <- !duplicated(keys)
  kept <- blocks[keep_idx]
  
  # Renumerar
  out <- character(0)
  if (length(prefix_lines)) out <- c(out, prefix_lines)
  
  for (k in seq_along(kept)) {
    b <- kept[[k]]
    b[1] <- sub("^\\s*\\d+\\.", paste0(k, "."), b[1])
    out <- c(out, b)
  }
  
  paste(out, collapse = "\n")
}

# ============================================================
# 4) ASK principal
#   - NEW: temperature
#   - Re-rank del set del candidato con Cross-Encoder
#   - FAIL-FAST: contexto SOLO del candidato (evita “mezcla”)
#   - PATCH: dedupe_numbered_list()
# ============================================================
ask <- function(candidate, question,
                k = 8L,
                openai_model = "gpt-4o-mini",
                openai_max_tokens = 700L,
                ollama_model = "llama3.2:3b",
                temperature = 0.2,
                print_console = TRUE) {
  
  candidate <- canonical_program(candidate)  # normaliza + valida
  
  # 1) embedding de la pregunta (para FAISS ancho)
  q_vec <- embed_one(question)
  q_np  <- np$expand_dims(np$array(q_vec, dtype = "float32"), 0L)
  
  # 2) cargar FAISS + chunks
  index <- faiss$read_index(INDEX_FP)
  chunks_tbl <- arrow::read_feather(CHUNKS_FP)
  
  # Asegurar tipos consistentes
  chunks_tbl <- chunks_tbl |>
    mutate(
      program = as.character(program),
      page_display = as.character(page_display),
      title = as.character(title),
      chunk = as.character(chunk)
    )
  
  # 3) búsqueda ancha (global)
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
  
  # 4) Filtra por candidato (fail-fast si queda vacío)
  topk_prog <- dplyr::filter(topk_global, program == candidate)
  
  # 5) Re-rank con Cross-Encoder sobre topk_prog (si existe)
  if (nrow(topk_prog) > 0) {
    ce_scores <- cross_rerank(question, topk_prog$chunk, batch_size = 32L)
    topk_prog <- topk_prog |>
      mutate(.ce_score = ce_scores) |>
      arrange(desc(.ce_score)) |>
      select(-.ce_score)
  }
  
  # Si aún hay menos de k, ampliamos con TODOS los chunks del candidato y re-rankeamos
  if (nrow(topk_prog) < k) {
    prog_all <- dplyr::filter(chunks_tbl, program == candidate)
    if (nrow(prog_all) == 0) {
      msg <- glue("No hay chunks para el programa de {candidate}.")
      message(msg)
      return(invisible(msg))
    }
    ce_scores_all <- cross_rerank(question, prog_all$chunk, batch_size = 32L)
    topk_prog <- prog_all |>
      mutate(.ce_score = ce_scores_all) |>
      arrange(desc(.ce_score)) |>
      select(-.ce_score)
  }
  
  topk_prog <- topk_prog[seq_len(min(k, nrow(topk_prog))), , drop = FALSE]
  if (nrow(topk_prog) == 0) {
    msg <- glue("No se menciona este tema en el programa de {candidate}.")
    message(msg)
    return(invisible(msg))
  }
  
  # FAIL-FAST: asegurar que NO hay mezcla antes de construir contexto
  if (!all(topk_prog$program == candidate)) {
    stop("Fail-fast: se detectaron chunks fuera del candidato seleccionado (mezcla de programas).")
  }
  
  # Contexto con páginas (manteniendo [[p.X]] para que el modelo elija X)
  pg_marker <- ifelse(is.na(topk_prog$page_display) | topk_prog$page_display == "",
                      "p.?", topk_prog$page_display)
  ctx_lines <- sprintf("[[%s]] %s", pg_marker, topk_prog$chunk)
  context_text <- paste(ctx_lines, collapse = "\n---\n")
  
  # ============================================================
  # PROMPT (PATCH: anti-markdown + sin introducciones + SOLO numerado + anti duplicados)
  # ============================================================
  prompt <- glue("
Responde en español DEVOLVIENDO SOLO una lista numerada con el formato exacto:
1. ...
2. ...
3. ...
(no agregues títulos, introducciones, contexto, ni texto fuera de la lista)

Formato OBLIGATORIO:
- Cada ítem debe terminar EXACTAMENTE con: (Programa {candidate}; Pág. X)
- Donde X es el número de página tomado del marcador [[p.X]] de los fragmentos usados.
- Si usas más de una página en el mismo ítem, usa: (Programa {candidate}; Págs. X, Y)

Prohibido:
- No uses markdown (no **negritas**, no encabezados, no sub-viñetas con '*', '-', '•').
- No incluyas los marcadores [[p.X]] en el texto final.
- No repitas ideas: si solo hay 1 punto sustentado por los fragmentos, devuelve SOLO 1 ítem (no dupliques).

Reglas:
- Usa SOLO información contenida en los fragmentos.
- No inventes ni completes con conocimiento externo.

Pregunta del usuario: {question}

Fragmentos recuperados (SOLO del Programa de {candidate}):
{context_text}
")
  # ============================================================
  # FIN PROMPT
  # ============================================================
  
  ans <- generate_answer(
    prompt = prompt,
    openai_model = openai_model,
    openai_max_tokens = openai_max_tokens,
    ollama_model = ollama_model,
    temperature = temperature
  )
  
  used_provider <- attr(ans, "used_provider") %||% "Desconocido"
  used_model    <- attr(ans, "used_model") %||% "Desconocido"
  
  # Fuerza citas por código (CE + borde de página)
  ans2 <- ensure_citations(ans, candidate, topk_prog, question_for_ce = question)
  
  # PATCH: elimina duplicados numerados + renumera
  ans2 <- dedupe_numbered_list(ans2)
  
  # Mantener metadata backend
  attr(ans2, "used_provider") <- used_provider
  attr(ans2, "used_model")    <- used_model
  
  if (isTRUE(print_console)) {
    cat("\n=====================\n")
    cat("Modelo usado: ", used_provider, " — ", used_model, "\n", sep = "")
    cat("Temperatura: ", format(temperature), "\n", sep = "")
    cat("Respuesta generada:\n")
    cat(ans2)
    cat("\n=====================\n")
  }
  
  invisible(ans2)
}

# Ejemplo (NO ejecutes si lo estás sourceando desde Shiny):
# ask("Parisi", "¿Qué medidas plantea en temas de seguridad e inmigración ilegal?", temperature = 0.15)
