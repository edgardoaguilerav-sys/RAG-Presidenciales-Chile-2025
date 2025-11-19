# ============================================================
# app.R — Consultas a Programas Presidenciales (UI mejorada)
# ============================================================

library(shiny)
library(shinythemes)

# Cargar la lógica RAG (define PROGRAMS y la función ask())
source("test_rag_cli.R")

# Lista de candidatos desde meta.json (ya cargado en test_rag_cli.R)
candidatos <- PROGRAMS

# ============================================================
# UI
# ============================================================

ui <- fluidPage(
  theme = shinytheme("flatly"),
  
  # ---- Estilos globales ----
  tags$head(
    tags$style(HTML("
      body {
        background-color: #f4f5f7;
      }
      .app-header {
        margin-bottom: 10px;
      }
      .app-title {
        font-weight: 700;
        font-size: 32px;
        margin-bottom: 5px;
      }
      .app-subtitle {
        color: #6c757d;
        font-size: 14px;
        margin-bottom: 25px;
      }
      .card-panel {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
      }
      .card-panel h4 {
        margin-top: 0;
      }
      #respuesta {
        white-space: pre-wrap;   /* respeta saltos de línea y hace wrap */
        word-wrap: break-word;   /* corta palabras largas si es necesario */
        max-height: 520px;       /* alto máximo del panel de respuesta */
        overflow-y: auto;        /* scroll vertical si es muy largo */
        overflow-x: hidden;      /* oculta el scroll horizontal */
        font-family: 'Courier New', monospace;
        font-size: 14px;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 14px;
      }
      .btn-primary {
        font-weight: 600;
        width: 100%;
      }
    "))
  ),
  
  # ---- Encabezado ----
  div(
    class = "app-header",
    h1(class = "app-title", "Consultas a Programas Presidenciales"),
    div(
      class = "app-subtitle",
      "Selecciona un candidato, formula tu pregunta y obtén una respuesta basada exclusivamente ",
      "en su programa de gobierno 2025."
    )
  ),
  
  # ---- Contenido principal ----
  fluidRow(
    
    # Panel izquierdo: filtros + pregunta
    column(
      width = 4,
      div(
        class = "card-panel",
        
        h4("Configuración de consulta"),
        tags$hr(),
        
        selectInput(
          inputId = "cand",
          label   = "Seleccione un candidato:",
          choices = candidatos
        ),
        
        textAreaInput(
          inputId    = "preg",
          label      = "Ingrese su pregunta:",
          placeholder = "Ejemplo: ¿Qué propone en materia de seguridad y combate al crimen organizado?",
          rows       = 5
        ),
        
        br(),
        actionButton(
          inputId = "go",
          label   = "Generar respuesta",
          class   = "btn btn-primary"
        ),
        
        br(), br(),
        tags$small(
          HTML("Nota: las respuestas se generan usando un modelo de lenguaje (OpenAI) y ",
               "fragmentos del programa seleccionado. No se utilizan fuentes externas.")
        )
      )
    ),
    
    # Panel derecho: respuesta
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

# ============================================================
# SERVER
# ============================================================

server <- function(input, output, session) {
  
  observeEvent(input$go, {
    req(input$cand)
    req(input$preg)
    
    output$respuesta <- renderText({
      tryCatch(
        {
          # Llama a tu función RAG (usa embeddings + FAISS + OpenAI)
          ask(input$cand, input$preg)
        },
        error = function(e) {
          paste("⚠️ Error al procesar la consulta:\n", e$message)
        }
      )
    })
  })
}

# ============================================================
# Lanzar app
# ============================================================

shinyApp(ui = ui, server = server)
