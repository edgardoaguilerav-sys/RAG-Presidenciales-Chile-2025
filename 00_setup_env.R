# 00_setup_env.R — preparar entorno Python "rag-faiss" con FAISS + bge-m3

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)

env_name <- "rag-faiss"

# ------------------------------------------------------------
# 1) Detectar / instalar Miniconda si no existe
# ------------------------------------------------------------
cb <- tryCatch(reticulate::conda_binary(), error = function(e) NULL)
if (is.null(cb)) {
  message("No se encontró conda. Instalando Miniconda (puede tardar unos minutos)…")
  reticulate::install_miniconda(update = TRUE)
  cb <- reticulate::conda_binary()
  if (is.null(cb)) stop("No se pudo instalar/detectar Miniconda.")
}

# cb apunta a .../r-miniconda/condabin/conda.bat → subimos dos niveles
miniconda_dir <- normalizePath(dirname(dirname(cb)), winslash = "/")

Sys.setenv(RETICULATE_MINICONDA_PATH = miniconda_dir)
Sys.setenv(PATH = paste(file.path(miniconda_dir, "Scripts"), Sys.getenv("PATH"), sep = ";"))

# ------------------------------------------------------------
# 2) Crear entorno rag-faiss si no existe
# ------------------------------------------------------------
cl <- reticulate::conda_list()
if (!env_name %in% cl$name) {
  message("Creando entorno conda '", env_name, "' …")
  cre <- system2(cb, c("create", "-y", "-n", env_name, "python=3.11",
                       "--override-channels", "-c", "conda-forge"))
  if (cre != 0) stop("No se pudo crear el entorno 'rag-faiss'.")
} else {
  message("Entorno '", env_name, "' ya existe; se reutilizará.")
}

# ------------------------------------------------------------
# 3) Ruta al python.exe del entorno
# ------------------------------------------------------------
py_exe <- file.path(miniconda_dir, "envs", env_name, "python.exe")
if (!file.exists(py_exe)) {
  stop("No se encontró python.exe del entorno 'rag-faiss' en: ", py_exe)
}

Sys.setenv(RETICULATE_PYTHON = py_exe)

# Activar Python en esta sesión
use_python(py_exe, required = TRUE)
py_config()

# ------------------------------------------------------------
# 4) Limpiar fastembed si estuviera instalado (ya no lo usamos)
# ------------------------------------------------------------
py_run_string("
import sys, subprocess
subprocess.call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'fastembed'])
")

# ------------------------------------------------------------
# 5) Instalar / actualizar paquetes principales vía pip
#    (FAISS + embeddings + trilogía HF consistente)
# ------------------------------------------------------------
py_install(
  c(
    'faiss-cpu',
    'numpy',
    'sentence-transformers',
    'transformers==4.57.1',
    'tokenizers>=0.22,<0.23',
    'huggingface-hub[hf_transfer]==0.36.0',
    'safetensors'
  ),
  envname = env_name,
  pip     = TRUE
)

# ------------------------------------------------------------
# 6) Torch CPU (vía índice oficial de PyTorch)
# ------------------------------------------------------------
py_install(
  'torch',
  envname     = env_name,
  pip         = TRUE,
  pip_options = '--index-url https://download.pytorch.org/whl/cpu --upgrade --force-reinstall --no-cache-dir'
)
py_install(
  'torchvision',
  envname     = env_name,
  pip         = TRUE,
  pip_options = '--index-url https://download.pytorch.org/whl/cpu --upgrade --force-reinstall --no-cache-dir'
)
py_install(
  'torchaudio',
  envname     = env_name,
  pip         = TRUE,
  pip_options = '--index-url https://download.pytorch.org/whl/cpu --upgrade --force-reinstall --no-cache-dir'
)

# ------------------------------------------------------------
# 7) Confirmar Torch
# ------------------------------------------------------------
py_run_string("import torch; print('TORCH', torch.__version__)")

cat("\n✅ Entorno listo. Python:", py_exe, " | Env:", env_name, "\n")
cat('   Puedes fijarlo en tu .Rprofile con:\n')
cat('   Sys.setenv(RETICULATE_PYTHON = \"', py_exe, '\")\n', sep = "")
