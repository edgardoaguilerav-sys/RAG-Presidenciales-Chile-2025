# ============================================================
# app.R — Consultas a Programas Presidenciales (UI mejorada)
# + feedback "Generando respuesta usando el modelo X"
# + deshabilitar botón mientras genera (shinyjs)
# ============================================================

library(shiny)
library(shinythemes)
library(shinyjs)
library(jsonlite)

source("test_rag_cli.R")

OUT_DIR <- "C:/Users/LENOVO/Desktop/RAG Programas Presidenciales/_rag_build_faiss"
META_FP <- file.path(OUT_DIR, "meta.json")

`%||%` <- function(x, y) if (!is.null(x)) x else y

candidatos <- NULL
if (exists("PROGRAMS", inherits = TRUE) && length(get("PROGRAMS", inherits = TRUE)) > 0) {
  candidatos <- sort(unique(get("PROGRAMS", inherits = TRUE)))
} else if (file.exists(META_FP)) {
  meta_app <- jsonlite::read_json(META_FP, simplifyVector = TRUE)
  candidatos <- sort(unique(meta_app$programs %||% character()))
}
if (is.null(candidatos) || length(candidatos) == 0) {
  candidatos <- c("Jara", "Kast", "Kayser", "Matthei", "Parisi")
}

ui <- fluidPage(
  theme = shinytheme("flatly"),
  useShinyjs(),
  
  tags$head(
    tags$style(HTML("
      body { background-color: #f4f5f7; }
      .app-header { margin-bottom: 10px; }
      .app-title { font-weight: 700; font-size: 32px; margin-bottom: 5px; }
      .app-subtitle { color: #6c757d; font-size: 14px; margin-bottom: 25px; }
      .card-panel {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
      }
      .card-panel h4 { margin-top: 0; }
      #respuesta {
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 520px;
        overflow-y: auto;
        overflow-x: hidden;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 14px;
      }
      .btn-primary { font-weight: 600; width: 100%; }
    "))
  ),
  
  div(
    class = "app-header",
    h1(class = "app-title", "Consultas a Programas Presidenciales"),
    div(
      class = "app-subtitle",
      "Selecciona un candidato, formula tu pregunta y obtén una respuesta basada exclusivamente ",
      "en su programa de gobierno 2025."
    )
  ),
  
  fluidRow(
    column(
      width = 4,
      div(
        class = "card-panel",
        h4("Configuración de consulta"),
        tags$hr(),
        
        selectInput(
          inputId = "cand",
          label   = "Seleccione un candidato:",
          choices = candidatos,
          selected = if ("Parisi" %in% candidatos) "Parisi" else candidatos[[1]]
        ),
        
        textAreaInput(
          inputId     = "preg",
          label       = "Ingrese su pregunta:",
          placeholder = "Ejemplo: ¿Qué propone en materia de seguridad y combate al crimen organizado?",
          rows        = 5
        ),
        
        br(),
        actionButton(
          inputId = "go",
          label   = "Generar respuesta",
          class   = "btn btn-primary"
        ),
        
        br(), br(),
        tags$small(
          HTML("Nota: las respuestas se generan usando un modelo de lenguaje (OpenAI u Ollama) y ",
               "fragmentos del programa seleccionado. No se utilizan fuentes externas.")
        )
      )
    ),
    
    column(
      width = 8,
      div(
        class = "card-panel",
        h4("Respuesta generada"),
        tags$hr(),
        verbatimTextOutput("respuesta", placeholder = TRUE)
      )
    )
  )
)

server <- function(input, output, session) {
  
  respuesta_val <- reactiveVal("")
  
  output$respuesta <- renderText({
    respuesta_val()
  })
  
  observeEvent(input$go, {
    req(input$cand)
    req(input$preg)
    
    # ---- Modelo esperado (solo para el mensaje mientras genera) ----
    openai_model_default <- "gpt-4o-mini"
    ollama_model_default <- "llama3.2:3b"
    
    using_openai <- nzchar(Sys.getenv("OPENAI_API_KEY", ""))
    model_label <- if (using_openai) {
      paste0("OpenAI: ", openai_model_default)
    } else {
      paste0("Ollama: ", ollama_model_default)
    }
    
    # Mensaje único (no redundante)
    msg <- paste0("Generando respuesta usando el modelo ", model_label, "…")
    
    shinyjs::disable("go")
    respuesta_val(msg)
    on.exit(shinyjs::enable("go"), add = TRUE)
    
    withProgress(message = msg, value = 0, {
      incProgress(0.2)  # sin detalle adicional (evita duplicar texto)
      Sys.sleep(0.05)
      
      out <- tryCatch({
        incProgress(0.6)
        ask(
          input$cand,
          input$preg,
          print_console = FALSE,
          openai_model = openai_model_default,
          ollama_model = ollama_model_default
        )
      }, error = function(e) {
        paste("⚠️ Error al procesar la consulta:\n", e$message)
      })
      
      incProgress(1)
      respuesta_val(out)
    })
  })
}

shinyApp(ui = ui, server = server)
