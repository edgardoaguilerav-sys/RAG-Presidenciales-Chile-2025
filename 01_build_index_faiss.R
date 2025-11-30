# ============================================================
# 01_build_index_faiss.R — Ingesta -> OCR si es necesario -> chunking -> embeddings -> FAISS
# Salidas: index.faiss, chunks.arrow, idmap.arrow, meta.json
# ============================================================

# === Rutas ===
DOC_DIR <- "C:/Users/LENOVO/Desktop/RAG Programas Presidenciales"
OUT_DIR <- file.path(DOC_DIR, "_rag_build_faiss")

# Forzar reconstrucción total del índice (recomendado cuando agregas PDFs)
FORCE_REBUILD <- TRUE

# ============================================================
# 0) Python/conda: detectar conda y apuntar al env rag-faiss (igual que en 00)
#    IMPORTANTE: si te da error, reinicia RStudio: Session -> Restart R
# ============================================================
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)

env_name <- "rag-faiss"

cb <- tryCatch(reticulate::conda_binary(), error = function(e) NULL)
if (is.null(cb)) stop("No se detectó conda. Ejecuta 00_setup_env.R primero.")

miniconda_dir <- normalizePath(dirname(dirname(cb)), winslash = "/")
py_exe <- file.path(miniconda_dir, "envs", env_name, "python.exe")

if (!file.exists(py_exe)) {
  stop("No existe python.exe del env '", env_name, "': ", py_exe,
       "\n(¿El env se creó en otra instalación de conda/miniconda?)")
}

Sys.setenv(RETICULATE_MINICONDA_PATH = miniconda_dir)
Sys.setenv(RETICULATE_PYTHON = py_exe)

use_python(py_exe, required = TRUE)
py_config()

# === Librerías R ===
libs <- c("pdftools","readtext","stringr","dplyr","purrr","tidyr","tibble",
          "jsonlite","arrow","glue","uuid","progress")
to_install <- libs[!libs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)
invisible(lapply(libs, library, character.only = TRUE))

# Paquetes para OCR (si un PDF viene como imagen)
need_ocr <- c("tesseract","png")
miss_ocr <- need_ocr[!need_ocr %in% rownames(installed.packages())]
if (length(miss_ocr)) install.packages(miss_ocr)

# === Salidas ===
if (FORCE_REBUILD && dir.exists(OUT_DIR)) {
  unlink(OUT_DIR, recursive = TRUE, force = TRUE)
}
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

CHUNKS_FP <- file.path(OUT_DIR, "chunks.arrow")
IDMAP_FP  <- file.path(OUT_DIR, "idmap.arrow")
FAISS_FP  <- file.path(OUT_DIR, "index.faiss")
META_FP   <- file.path(OUT_DIR, "meta.json")

# === Parámetros de chunking ===
TARGET_TOK <- 450L
OVERLAP    <- 80L

# === Modelo embeddings (sentence-transformers; CPU Torch) ===
LOCAL_EMB_MODEL <- "intfloat/multilingual-e5-large"  # 1024 dims

# === Etiquetado de programas (robusto por regex) ===
program_patterns <- tibble::tribble(
  ~pattern,                     ~label,
  "(?i)jara",                   "Jara",
  "(?i)matthei|mattei",         "Matthei",
  "(?i)kayser|kaiser",          "Kayser",
  "(?i)kast",                   "Kast",
  "(?i)parisi",                 "Parisi"
)

detect_program <- function(base_name) {
  hit <- program_patterns |>
    dplyr::filter(stringr::str_detect(base_name, pattern)) |>
    dplyr::slice(1)
  if (nrow(hit)) hit$label[[1]] else base_name
}

# ---------- Helpers ----------
split_tokens <- function(x) unlist(strsplit(x, "(?<=\\s)", perl = TRUE))

chunk_text <- function(txt, target = TARGET_TOK, overlap = OVERLAP) {
  toks <- split_tokens(txt)
  n <- length(toks)
  if (!n) return(character())
  starts <- seq(1, n, by = target - overlap)
  purrr::map_chr(starts, function(s) {
    e <- min(n, s + target - 1)
    paste0(toks[s:e], collapse = "")
  })
}

read_any <- function(path) {
  ext <- tolower(tools::file_ext(path))
  
  if (ext %in% c("txt","md","csv","tsv","r","py")) {
    return(tibble::tibble(
      page = NA_integer_,
      text = paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
    ))
  }
  
  if (ext == "pdf") {
    # 1) Intento normal
    pgs <- pdftools::pdf_text(path)
    txt <- paste(pgs, collapse = "\n")
    
    # 2) Fallback OCR si hay muy poco texto
    if (nchar(gsub("\\s+", "", txt)) < 500) {
      message("📄 OCR activado para: ", basename(path))
      eng <- tesseract::tesseract("spa")
      
      info <- pdftools::pdf_info(path)
      n <- info$pages
      if (is.null(n) || n < 1) return(tibble::tibble(page = integer(), text = character()))
      
      tmpdir <- tempfile("ocr_"); dir.create(tmpdir)
      png_files <- pdftools::pdf_convert(
        pdf       = path,
        format    = "png",
        pages     = 1:n,
        filenames = file.path(tmpdir, sprintf("page_%03d.png", 1:n)),
        dpi       = 220,
        verbose   = FALSE
      )
      
      ocr_pages <- vapply(png_files, function(pngfp) {
        tryCatch(tesseract::ocr(pngfp, engine = eng), error = function(e) "")
      }, FUN.VALUE = character(1))
      
      unlink(tmpdir, recursive = TRUE, force = TRUE)
      
      if (all(nchar(gsub("\\s+", "", ocr_pages)) == 0)) {
        return(tibble::tibble(page = integer(), text = character()))
      }
      
      pgs <- ocr_pages
    }
    
    return(tibble::tibble(page = seq_along(pgs), text = pgs))
  }
  
  if (ext %in% c("docx","rtf","doc")) {
    rt <- readtext::readtext(path)
    return(tibble::tibble(page = NA_integer_, text = paste(rt$text, collapse = "\n")))
  }
  
  message("⚠️ Omitiendo formato no soportado: ", basename(path))
  tibble::tibble(page = integer(), text = character())
}

# ---------- Ingesta + chunking (incluye subcarpetas, excluye OUT_DIR) ----------
files <- list.files(
  DOC_DIR, full.names = TRUE, recursive = TRUE,
  pattern = "\\.(pdf|docx|rtf|doc|txt|md)$", ignore.case = TRUE
)
stopifnot(length(files) > 0)

# excluir todo lo que esté dentro de OUT_DIR
out_norm <- normalizePath(OUT_DIR, winslash = "/", mustWork = FALSE)
files <- files[!startsWith(normalizePath(files, winslash = "/", mustWork = FALSE), out_norm)]

# excluir README.* (evita que aparezca como "programa")
files <- files[!grepl("^README(\\.|$)", basename(files), ignore.case = TRUE)]

stopifnot(length(files) > 0)

rows <- list()
for (f in files) {
  base <- tools::file_path_sans_ext(basename(f))
  program <- detect_program(base)
  
  df <- read_any(f)
  if (!nrow(df)) next
  
  df <- df |>
    dplyr::mutate(
      file = f,
      title = base,
      text  = stringr::str_replace_all(text, "\r", "\n") |> stringr::str_squish(),
      program = program
    )
  
  chs <- purrr::pmap_dfr(df, \(page, text, file, title, program){
    cs <- chunk_text(text)
    if (!length(cs)) return(tibble::tibble())
    tibble::tibble(
      chunk_id = paste0(uuid::UUIDgenerate(), "-", seq_along(cs)),
      file, title, program,
      page = page,
      page_display = ifelse(is.na(page), "s/p", paste0("p.", page)),
      chunk = cs
    )
  })
  
  rows[[length(rows)+1]] <- chs
}

chunks <- dplyr::bind_rows(rows)
stopifnot(nrow(chunks) > 0)

cat(glue::glue("\n📚 Documentos detectados: {length(unique(chunks$file))}\n"))
cat(glue::glue("🏷️ Programas: {paste(sort(unique(chunks$program)), collapse = ', ')}\n\n"))

# ---------- Python (SentenceTransformers + FAISS) ----------
faiss <- reticulate::import("faiss")
np    <- reticulate::import("numpy")
st    <- reticulate::import("sentence_transformers")

embedder <- st$SentenceTransformer(LOCAL_EMB_MODEL, device = "cpu")

cat(glue::glue("Generando embeddings (SentenceTransformers: {LOCAL_EMB_MODEL}) para {nrow(chunks)} chunks...\n"))

BATCH <- 64L
emb_list <- list()
pb <- progress::progress_bar$new(
  total = nrow(chunks),
  format = "[:bar] :percent :current/:total ETA: :eta"
)

for (i in seq(1, nrow(chunks), by = BATCH)) {
  j <- min(i + BATCH - 1L, nrow(chunks))
  texts_batch <- chunks$chunk[i:j]
  
  # normalize_embeddings=TRUE => listo para cosine via inner-product
  emb_batch <- embedder$encode(
    texts_batch,
    batch_size = as.integer(length(texts_batch)),
    show_progress_bar = FALSE,
    convert_to_numpy = TRUE,
    normalize_embeddings = TRUE
  )
  
  emb_list[[length(emb_list)+1]] <- as.matrix(emb_batch)
  pb$tick(j - i + 1L)
}

emb <- do.call(rbind, emb_list)
storage.mode(emb) <- "double"
D <- ncol(emb)

# ---------- FAISS (cosine via Inner Product) ----------
emb32 <- np$array(emb, dtype = "float32")
index <- faiss$IndexFlatIP(as.integer(D))
index$add(emb32)
faiss$write_index(index, FAISS_FP)

# ---------- Persistencia ----------
chunks_to_save <- chunks |>
  dplyr::mutate(row_id = dplyr::row_number()) |>
  dplyr::select(row_id, chunk_id, program, file, title, page, page_display, chunk)
arrow::write_feather(chunks_to_save, CHUNKS_FP)

idmap <- tibble::tibble(faiss_id0 = 0:(nrow(chunks_to_save)-1L), row_id = chunks_to_save$row_id)
arrow::write_feather(idmap, IDMAP_FP)

meta <- list(
  doc_dir  = DOC_DIR,
  out_dir  = OUT_DIR,
  faiss    = list(file = FAISS_FP, type = "flatip", n_items = nrow(chunks_to_save)),
  idmap    = IDMAP_FP,
  chunks   = CHUNKS_FP,
  emb      = list(provider = "sentence-transformers", local_model = LOCAL_EMB_MODEL, dim = D, normalized = TRUE),
  programs = sort(unique(chunks_to_save$program)),
  built_at = as.character(Sys.time())
)
jsonlite::write_json(meta, META_FP, pretty = TRUE, auto_unbox = TRUE)

cat("\n✅ Indexación FAISS completada.\n")
cat(glue::glue("- Índice: {FAISS_FP}\n- Chunks: {CHUNKS_FP}\n- IdMap: {IDMAP_FP}\n- Meta: {META_FP}\n"))
