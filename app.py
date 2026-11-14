"""
Actividad 12 - Chatbot con Streamlit y OpenAI
Chatbot que responde preguntas de texto libre o sobre una tabla CSV subida por el usuario.
Si hay un CSV cargado, el bot SOLO responde preguntas relacionadas con esos datos.

Modelo: gpt-4o-mini (OpenAI).
"""

import os
import re
import unicodedata

import pandas as pd
import streamlit as st
from openai import OpenAI

MODEL = "gpt-4o-mini"
SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "mall_customers.csv")
MAX_CSV_CHARS = 60_000  # si el CSV es más grande, se envían solo resúmenes + primeras filas
OUT_OF_SCOPE_MSG = (
    "Lo siento, solo puedo responder preguntas relacionadas con el dataset cargado. "
    "Por favor, reformula tu pregunta sobre los datos."
)

st.set_page_config(page_title="Chatbot de Datos", page_icon="🤖", layout="centered")


# ---------------------------------------------------------------------------
# API key: primero st.secrets (Streamlit Cloud), luego variable de entorno,
# y por último un campo en la barra lateral.
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass  # no existe secrets.toml en local
    return os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Procesamiento del CSV
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def build_data_context(df: pd.DataFrame) -> str:
    """Convierte el DataFrame en texto + resúmenes calculados con pandas
    (así las cifras del modelo son exactas)."""
    parts = [
        f"Filas: {len(df)} | Columnas: {len(df.columns)}",
        "Columnas y tipos: " + ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items()),
        "Estadísticas descriptivas:\n" + df.describe().round(2).to_string(),
    ]

    numeric = [c for c in df.select_dtypes("number").columns if "id" not in c.lower()]
    categorical = [c for c in df.select_dtypes(exclude="number").columns if df[c].nunique() <= 20]

    for c in categorical:
        parts.append(f"Conteo por {c}:\n" + df[c].value_counts().to_string())
        if numeric:
            parts.append(f"Promedios por {c}:\n" + df.groupby(c)[numeric].mean().round(2).to_string())

    age_col = next((c for c in df.columns if c.lower() in ("age", "edad")), None)
    if age_col and numeric:
        bins = [0, 25, 35, 50, 200]
        labels = ["<25", "25-34", "35-49", "50+"]
        groups = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
        grouped = df.groupby(groups, observed=True)
        summary = grouped[numeric].mean().round(2).add_prefix("Promedio ")
        summary.insert(0, "Clientes", grouped.size())
        summary.index.name = f"Grupo de {age_col}"
        parts.append(f"Resumen por grupo de {age_col} (cada fila es un grupo):\n" + summary.to_string())

    csv_text = df.to_csv(index=False)
    if len(csv_text) <= MAX_CSV_CHARS:
        parts.append("Datos completos (CSV):\n" + csv_text)
    else:
        parts.append("Primeras 20 filas (el dataset es grande):\n" + df.head(20).to_csv(index=False))
    return "\n\n".join(parts)


def system_prompt(df: pd.DataFrame | None) -> str:
    if df is None:
        return "Eres un asistente útil. Responde en español, de forma clara y breve."
    return (
        "Eres un analista de datos. Solo respondes preguntas sobre el dataset de abajo "
        "(estadísticas, segmentos, tendencias y recomendaciones de marketing basadas en estos datos). "
        "Usa las cifras ya calculadas; no inventes datos. Responde en español, de forma clara y breve.\n"
        f'Si la pregunta no trata sobre el dataset, responde solo: "{OUT_OF_SCOPE_MSG}"\n\n'
        "=== DATASET ===\n" + build_data_context(df)
    )


# ---------------------------------------------------------------------------
# Filtro de alcance: además de la instrucción al modelo, revisamos antes si la
# pregunta tiene relación con los datos (así el rechazo es seguro y no gasta tokens).
# ---------------------------------------------------------------------------
DATA_STEMS = [
    # español
    "client", "edad", "genero", "sexo", "hombre", "mujer", "ingreso", "salario", "gast", "puntu",
    "segment", "marketing", "campan", "publicidad", "anuncio", "canal", "promedi", "media", "median",
    "total", "cuant", "porcentaj", "distribu", "dato", "tabla", "fila", "columna", "grupo", "agrup",
    "correlac", "maxim", "minim", "resum", "estadist", "analiz", "analisis", "compar", "tendenc",
    "perfil", "joven", "jovenes", "mayor", "menor", "rango", "registro", "csv", "dataset", "patron",
    # inglés
    "customer", "age", "gender", "male", "female", "income", "spend", "score", "average", "mean",
    "how many", "percent", "data", "table", "row", "column", "group", "correlat", "summar", "statist",
    "analy", "trend", "profile", "young", "older", "record",
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def is_about_data(question: str, df: pd.DataFrame) -> bool:
    q = normalize(question)
    words = re.findall(r"[a-z0-9]+", q)
    column_words = {w for c in df.columns for w in re.findall(r"[a-z0-9]+", normalize(c)) if len(w) >= 3}
    stems = set(DATA_STEMS) | column_words
    for stem in stems:
        if " " in stem:
            if stem in q:
                return True
        elif any(w.startswith(stem) for w in words):
            return True
    return False


# ---------------------------------------------------------------------------
# Barra lateral: carga del CSV
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("OpenAI API key", type="password",
                                help="No se guarda; solo se usa durante esta sesión.")

    st.divider()
    st.header("📄 Datos")
    uploaded_file = st.file_uploader("Sube una tabla (CSV)", type=["csv"])
    use_sample = False
    if uploaded_file is None and os.path.exists(SAMPLE_CSV):
        use_sample = st.checkbox("Usar dataset de ejemplo (mall_customers.csv)")

    if st.button("🗑️ Borrar conversación"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Modelo: `{MODEL}` (OpenAI)")

df, data_name = None, None
try:
    if uploaded_file is not None:
        df, data_name = load_csv(uploaded_file), uploaded_file.name
    elif use_sample:
        df, data_name = load_csv(SAMPLE_CSV), "mall_customers.csv"
except Exception as e:
    st.sidebar.error(f"No se pudo leer el CSV: {e}")


# ---------------------------------------------------------------------------
# Interfaz principal
# ---------------------------------------------------------------------------
st.title("🤖 Chatbot de Datos")

if df is not None:
    st.success(f"Dataset cargado: **{data_name}** ({len(df)} filas, {len(df.columns)} columnas). "
               "El chatbot solo responderá preguntas sobre estos datos.")
    with st.expander("Vista previa de los datos"):
        st.dataframe(df.head(20), width="stretch")
else:
    st.info("Escribe cualquier pregunta, o sube un CSV en la barra lateral para hacer preguntas sobre tus datos.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Escribe tu pregunta...")

if prompt:
    if not api_key:
        st.warning("Ingresa tu OpenAI API key en la barra lateral para continuar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if df is not None and not is_about_data(prompt, df):
            answer = OUT_OF_SCOPE_MSG
            st.markdown(answer)
        else:
            history = st.session_state.messages[-10:]
            messages = [{"role": "system", "content": system_prompt(df)}] + history
            try:
                client = OpenAI(api_key=api_key)
                stream = client.chat.completions.create(
                    model=MODEL, messages=messages, temperature=0.3, stream=True,
                )
                answer = st.write_stream(stream)
            except Exception as e:
                answer = f"⚠️ Error al llamar a la API de OpenAI: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
