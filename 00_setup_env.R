# ============================================================
# 00_setup_env.R — preparar entorno Python "rag-faiss"
# FAISS + HuggingFace + Torch CPU estable
# SIN fastembed (ya no se usa)
# ============================================================

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)

env_name <- "rag-faiss"

# ------------------------------------------------------------
# 1) Detectar / instalar Miniconda si no existe
# ------------------------------------------------------------
cb <- tryCatch(reticulate::conda_binary(), error = function(e) NULL)

if (is.null(cb)) {
  message("No se encontró conda. Instalando Miniconda…")
  reticulate::install_miniconda(update = TRUE)
  cb <- reticulate::conda_binary()
  if (is.null(cb)) stop("No se pudo instalar/detectar Miniconda.")
}

# Ruta base miniconda
miniconda_dir <- normalizePath(dirname(dirname(cb)), winslash = "/")

Sys.setenv(RETICULATE_MINICONDA_PATH = miniconda_dir)
Sys.setenv(PATH = paste(file.path(miniconda_dir, "Scripts"), Sys.getenv("PATH"), sep = ";"))

# ------------------------------------------------------------
# 2) Crear entorno rag-faiss si no existe
# ------------------------------------------------------------
cl <- reticulate::conda_list()

if (!env_name %in% cl$name) {
  message("Creando entorno conda '", env_name, "' …")
  cre <- system2(
    cb,
    c("create", "-y", "-n", env_name, "python=3.11",
      "--override-channels", "-c", "conda-forge")
  )
  if (cre != 0) stop("No se pudo crear el entorno 'rag-faiss'.")
} else {
  message("Entorno '", env_name, "' ya existe; se reutilizará.")
}

# ------------------------------------------------------------
# 3) Ruta a python.exe del entorno
# ------------------------------------------------------------
py_exe <- file.path(miniconda_dir, "envs", env_name, "python.exe")

if (!file.exists(py_exe)) {
  stop("No se encontró python.exe del entorno 'rag-faiss': ", py_exe)
}

Sys.setenv(RETICULATE_PYTHON = py_exe)
use_python(py_exe, required = TRUE)
py_config()

# 4) Limpiar fastembed si estuviera instalado (forma segura en Windows)
if (py_module_available("fastembed")) {
  message("Desinstalando fastembed…")
  try(py_install("fastembed", envname = env_name, pip = TRUE, pip_options = "--no-cache-dir -U --force-reinstall", uninstall = TRUE), silent = TRUE)
}


# ------------------------------------------------------------
# 5) Instalar paquetes principales vía pip
# ------------------------------------------------------------
py_install(
  packages = c(
    "faiss-cpu",
    "numpy",
    "transformers==4.57.1",
    "sentence-transformers",
    "huggingface-hub[hf_transfer]==0.36.0",
    "tokenizers>=0.22,<0.23",
    "safetensors"
  ),
  envname  = env_name,
  pip      = TRUE
)

# ------------------------------------------------------------
# 6) Instalar Torch CPU (versión estable)
# ------------------------------------------------------------
torch_url <- "https://download.pytorch.org/whl/cpu"

py_install(
  "torch",
  envname     = env_name,
  pip         = TRUE,
  pip_options = paste("--index-url", torch_url, "--upgrade --force-reinstall --no-cache-dir")
)

py_install(
  "torchvision",
  envname     = env_name,
  pip         = TRUE,
  pip_options = paste("--index-url", torch_url, "--upgrade --force-reinstall --no-cache-dir")
)

py_install(
  "torchaudio",
  envname     = env_name,
  pip         = TRUE,
  pip_options = paste("--index-url", torch_url, "--upgrade --force-reinstall --no-cache-dir")
)

# ------------------------------------------------------------
# 7) Confirmar Torch
# ------------------------------------------------------------
py_run_string("import torch; print('TORCH OK ->', torch.__version__)")

cat("\n============================================================\n")
cat("✅ Entorno 'rag-faiss' listo y operativo.\n")
cat("Python:", py_exe, "\n")
cat("Puedes fijarlo en tu .Rprofile con:\n")
cat("Sys.setenv(RETICULATE_PYTHON = \"", py_exe, "\")\n", sep = "")
cat("============================================================\n")
