# Actividad 12 – Chatbot de Datos con Streamlit y OpenAI

**App publicada:** https://ENLACE.streamlit.app

## Descripción
Desarrollé un chatbot web con Streamlit que usa el modelo `gpt-4o-mini` de OpenAI para responder preguntas en lenguaje natural sobre una tabla CSV.

- El usuario puede subir su propio CSV o usar el dataset de ejemplo `mall_customers.csv` (Kaggle: *Mall Customer Segmentation Data*, 200 clientes con género, edad, ingreso anual y puntuación de gasto).
- Cuando hay un CSV cargado, el chatbot **solo** responde preguntas relacionadas con esos datos; cualquier otra pregunta se rechaza con un mensaje fijo.
- Sin CSV, funciona como asistente de preguntas libres.

## Funcionamiento
- **Contexto de datos:** con pandas se calculan las estadísticas descriptivas, conteos y promedios por categoría y un resumen por grupos de edad (<25, 25-34, 35-49, 50+). Estas cifras se envían al modelo junto con los datos, de modo que las respuestas usan valores exactos y no inventados.
- **Control de alcance en dos niveles:**
  1. Un filtro previo revisa si la pregunta menciona términos del dataset (columnas, estadísticas, segmentos, marketing). Si no, se rechaza sin llamar a la API.
  2. El prompt de sistema le indica al modelo que rechace cualquier pregunta ajena a los datos.
- **API key:** se lee desde los *Secrets* de Streamlit Cloud, por lo que no está en el código ni en el repositorio.
- **Interfaz:** chat con historial, respuestas en streaming, vista previa de los datos y botón para borrar la conversación.

## Archivos
| Archivo | Contenido |
|---|---|
| `app.py` | Aplicación Streamlit (chat, procesamiento del CSV y filtro de alcance) |
| `mall_customers.csv` | Dataset de ejemplo |
| `requirements.txt` | Dependencias: streamlit, openai, pandas |
| `OpenAI_API.ipynb` | Notebook de práctica con la API de OpenAI |

## Pruebas realizadas (dataset de ejemplo)
| Prueba | Pregunta | Resultado |
|---|---|---|
| Respuesta útil | "Divide a los clientes por edad. ¿Qué canales de marketing serían más efectivos para cada grupo?" | Segmenta a los clientes en 4 grupos de edad con su ingreso y gasto promedio y recomienda canales para cada uno (redes sociales e influencers para los jóvenes, medios tradicionales para los mayores de 50). |
| Respuesta útil | "¿Cuál es el ingreso anual promedio por género?" | Mujeres: 59.25 k$ · Hombres: 62.23 k$ (coincide con el cálculo en pandas). |
| Fuera de alcance | "¿Cuál es la capital de Francia?" | "Lo siento, solo puedo responder preguntas relacionadas con el dataset cargado. Por favor, reformula tu pregunta sobre los datos." |
| Fuera de alcance | "Escríbeme una receta de pizza" | El mismo mensaje de rechazo. |
