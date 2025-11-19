# 01_build_index_faiss.R — Ingesta -> OCR si es necesario -> chunking -> FastEmbed -> FAISS
# Salidas: index.faiss, chunks.arrow, idmap.arrow, meta.json

# === Rutas ===
DOC_DIR <- "C:/Users/LENOVO/Desktop/RAG Programas Presidenciales"
OUT_DIR <- file.path(DOC_DIR, "_rag_build_faiss")

# Fijar el Python del entorno (ajusta si tu ruta difiere)
Sys.setenv(RETICULATE_PYTHON = "C:/Users/LENOVO/miniconda3/envs/rag-faiss/python.exe")

# === Librerías R ===
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
libs <- c("pdftools","readtext","stringr","dplyr","purrr","tidyr","tibble",
          "jsonlite","arrow","glue","uuid","progress")
to_install <- libs[!libs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)
invisible(lapply(libs, library, character.only = TRUE))
library(reticulate)

# Paquetes para OCR (si un PDF viene como imagen)
need_ocr <- c("tesseract","png")
miss_ocr <- need_ocr[!need_ocr %in% rownames(installed.packages())]
if (length(miss_ocr)) install.packages(miss_ocr)

# === Salidas ===
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
CHUNKS_FP <- file.path(OUT_DIR, "chunks.arrow")
IDMAP_FP  <- file.path(OUT_DIR, "idmap.arrow")
FAISS_FP  <- file.path(OUT_DIR, "index.faiss")
META_FP   <- file.path(OUT_DIR, "meta.json")

# === Parámetros de chunking ===
TARGET_TOK <- 450L
OVERLAP    <- 80L

# === Modelo de embeddings (FastEmbed, multilingüe, sin Torch) ===
LOCAL_EMB_MODEL <- "intfloat/multilingual-e5-large"  # 1024 dims, español muy bien

# === Mapeo de nombres archivo -> etiqueta de programa (para dropdown) ===
program_map <- c(
  "Programa_Jara"    = "Jara",
  "Programa_Matthei" = "Matthei",
  "Programa_Kayser"  = "Kayser",
  "Programa_Kast"    = "Kast"
)

# ---------- Helpers ----------
split_tokens <- function(x) {
  unlist(strsplit(x, "(?<=\\s)", perl = TRUE))
}

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

l2_normalize <- function(M) {
  n <- sqrt(rowSums(M^2))
  n[n == 0] <- 1
  M / n
}


# ---------- Helpers ----------
read_any <- function(path) {
  ext <- tolower(tools::file_ext(path))
  
  if (ext %in% c("txt","md","csv","tsv","r","py")) {
    return(tibble(page = NA_integer_,
                  text = paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")))
  }
  
  if (ext == "pdf") {
    # 1) Intento normal
    pgs <- pdftools::pdf_text(path)
    txt <- paste(pgs, collapse = "\n")
    
    # 2) Fallback OCR si hay muy poco texto
    if (nchar(gsub("\\s+", "", txt)) < 500) {
      message("📄 OCR activado para: ", basename(path))
      if (!requireNamespace("tesseract", quietly = TRUE))
        stop("Falta el paquete 'tesseract'. Instálalo e intenta de nuevo.")
      eng <- tesseract::tesseract("spa")
      
      info <- pdftools::pdf_info(path)
      n <- info$pages
      if (is.null(n) || n < 1) return(tibble(page = integer(), text = character()))
      
      # Usa pdf_convert para generar PNGs válidos por página (evita el problema de raw/255)
      tmpdir <- tempfile("ocr_"); dir.create(tmpdir)
      png_files <- pdftools::pdf_convert(
        pdf       = path,
        format    = "png",
        pages     = 1:n,
        filenames = file.path(tmpdir, sprintf("page_%03d.png", 1:n)),
        dpi       = 220,
        verbose   = FALSE
      )
      
      # OCR por página con manejo de errores
      ocr_pages <- vapply(png_files, function(pngfp) {
        out <- tryCatch(tesseract::ocr(pngfp, engine = eng), error = function(e) "")
        out
      }, FUN.VALUE = character(1))
      
      # Limpieza temporal
      unlink(tmpdir, recursive = TRUE, force = TRUE)
      
      # Si el OCR no devolvió nada útil, devolvemos vacío para que el caller lo omita
      if (all(nchar(gsub("\\s+", "", ocr_pages)) == 0))
        return(tibble(page = integer(), text = character()))
      
      pgs <- ocr_pages
    }
    
    return(tibble(page = seq_along(pgs), text = pgs))
  }
  
  if (ext %in% c("docx","rtf","doc")) {
    rt <- readtext::readtext(path)
    return(tibble(page = NA_integer_, text = paste(rt$text, collapse = "\n")))
  }
  
  message("⚠️ Omitiendo formato no soportado: ", basename(path))
  tibble(page = integer(), text = character())
}

# ---------- Ingesta + chunking ----------
files <- list.files(DOC_DIR, full.names = TRUE, recursive = FALSE,
                    pattern = "\\.(pdf|docx|rtf|doc|txt|md)$", ignore.case = TRUE)
stopifnot(length(files) > 0)

rows <- list()
for (f in files) {
  base <- tools::file_path_sans_ext(basename(f))
  hit  <- names(program_map)[stringr::str_detect(base, names(program_map))]
  program <- if (length(hit)) program_map[[hit[1]]] else base
  
  df <- read_any(f)
  if (!nrow(df)) next
  df <- df |>
    mutate(file = f,
           title = tools::file_path_sans_ext(basename(f)),
           text  = stringr::str_replace_all(text, "\r", "\n") |> stringr::str_squish(),
           program = program)
  
  chs <- purrr::pmap_dfr(df, \(page, text, file, title, program){
    cs <- chunk_text(text)
    if (!length(cs)) return(tibble())
    tibble(
      chunk_id = paste0(UUIDgenerate(), "-", seq_along(cs)),
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

# ---------- Python (FastEmbed + FAISS) ----------
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
py_config()

# Instalar fastembed/onnxruntime si faltaran (por seguridad)
if (!py_module_available("fastembed")) {
  py_install(c("onnxruntime>=1.17.0","fastembed>=0.3.3"), pip = TRUE)
}
faiss <- import("faiss")
fe    <- import("fastembed")
np    <- import("numpy")

# Instanciar el embedder (sin Torch)
embedder <- fe$TextEmbedding(model_name = LOCAL_EMB_MODEL)

cat(glue("\nGenerando embeddings (FastEmbed: {LOCAL_EMB_MODEL}) para {nrow(chunks)} chunks...\n"))

BATCH <- 64L
emb_list <- list()
pb <- progress_bar$new(total = nrow(chunks), format = "[:bar] :percent :current/:total ETA: :eta")

for (i in seq(1, nrow(chunks), by = BATCH)) {
  j <- min(i + BATCH - 1L, nrow(chunks))
  
  # Puente R->Python vía r$ (evita py_assign)
  texts_batch <- chunks$chunk[i:j]
  res <- py_run_string("
vecs = list(r.embedder.embed(r.texts_batch))
")
  vecs <- do.call(rbind, lapply(res$vecs, as.numeric))
  emb_list[[length(emb_list)+1]] <- vecs
  pb$tick(j - i + 1L)
}

emb <- do.call(rbind, emb_list)
storage.mode(emb) <- "double"

# Normalizar para coseno (Inner Product)
emb <- l2_normalize(emb)
D <- ncol(emb)

# ---------- FAISS (cosine via Inner Product) ----------
emb32 <- np$array(emb, dtype = "float32")
index <- faiss$IndexFlatIP(as.integer(D))
index$add(emb32)
faiss$write_index(index, FAISS_FP)

# ---------- Persistencia ----------
chunks_to_save <- chunks |>
  mutate(row_id = dplyr::row_number()) |>
  select(row_id, chunk_id, program, file, title, page, page_display, chunk)
arrow::write_feather(chunks_to_save, CHUNKS_FP)

idmap <- tibble(faiss_id0 = 0:(nrow(chunks_to_save)-1L), row_id = chunks_to_save$row_id)
arrow::write_feather(idmap, IDMAP_FP)

meta <- list(
  doc_dir  = DOC_DIR,
  out_dir  = OUT_DIR,
  faiss    = list(file = FAISS_FP, type = "flatip", n_items = nrow(chunks_to_save)),
  idmap    = IDMAP_FP,
  chunks   = CHUNKS_FP,
  emb      = list(provider = "local-fastembed", local_model = LOCAL_EMB_MODEL, dim = D, normalized = TRUE),
  programs = sort(unique(chunks_to_save$program)),
  built_at = as.character(Sys.time())
)
jsonlite::write_json(meta, META_FP, pretty = TRUE, auto_unbox = TRUE)

cat("\n✅ Indexación FAISS completada.\n")
cat(glue("- Índice: {FAISS_FP}\n- Chunks: {CHUNKS_FP}\n- IdMap: {IDMAP_FP}\n- Meta: {META_FP}\n"))
