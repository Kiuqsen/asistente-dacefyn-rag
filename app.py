"""
------ El Archivo appy.py-------------------------------------------------------------------------------------------
 Este script es el Frontend y Motor de Consultas RAG. La Interfaz es construida con Streamlit. Contiene las reglas de System Prompting y control de alucinaciones para el Asistente del DACEFyN.

 --- EXPLICACIÓN DE LIBRERÍAS ---

 streamlit: El framework para la interfaz (FRONTEND). Permite construir una aplicación web interactiva usando puro Python, ideal para prototipar rápido sin tener que programar en HTML, CSS o JavaScript.

 ------------------------------- BASE DE DATOS VECTORIAL ---
 chromadb: Es la librería principal de nuestra base de datos local. Nos permite abrir la carpeta física ('./chroma_db') donde guardamos los vectores previamente procesados y hacer consultas sobre ella.

 # ----------------------------- NÚCLEO DE LLAMAINDEX (El "Cerebro" del RAG) ---
 # VectorStoreIndex: Es el objeto principal que usamos para "envolver" nuestra base de datos y convertirla 
 # en un motor de búsqueda (query_engine) al que le podemos hacer preguntas.
 # Settings: Es el panel de control global de nuestra app. Aquí definimos qué "cerebro" (LLM) y qué 
 # "traductor" (Embeddings) va a usar todo el sistema por defecto.

 # ----------------------------- EL PUENTE DE CONEXIÓN ---
 # ChromaVectorStore: LlamaIndex no sabe hablar con ChromaDB de forma nativa. Esta clase actúa como 
 # un "driver" o traductor para que LlamaIndex pueda leer los datos que guardamos en ChromaDB.

 # ----------------------------- CONECTORES DE MODELOS (Vía LM Studio) ---
 # OpenAIEmbedding: Se encarga de transformar la pregunta del usuario en un vector matemático para poder buscar similitudes en la base de datos. (Apunta a al modelo 'paraphrase-multilingual' en LM Studio).
 # OpenAILike: Es el conector para nuestro Modelo de Lenguaje (LLM) principal. Se llama "OpenAILike" porque se conecta a servidores locales (como LM Studio) que imitan la forma en que se comunica ChatGPT.

 # ----------------------------- HERRAMIENTAS DE INGENIERÍA DE PROMPTS (El sistema Anti-Alucinaciones) ---
 # ChatPromptTemplate: Nos permite crear una "plantilla" fija para las preguntas. En lugar de mandar la pregunta del usuario suelta, la envuelve con nuestras reglas estrictas antes de enviarla al modelo.
 # ChatMessage, MessageRole: Son las herramientas para definir "quién dice qué" en la plantilla. 
 # MessageRole nos permite separar claramente qué instrucciones son del SISTEMA (las reglas absolutas del SYSTEM PROMPTING programado, como "No inventes nada") y qué es del USUARIO (la pregunta real).
"""

import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

# --- 1. CONFIGURACIÓN DEL MOTOR RAG ---
# Igual que en el archivo anterior (crear_base_datos.py), conectamos con LM Studio.
Settings.embed_model = OpenAIEmbedding(
    model_name="paraphrase-multilingual-minilm-l12-v2.gguf",
    api_base="http://localhost:1234/v1",
    api_key="not-needed"
)


# Conexión con LM Studio para LLM (Optimizado para Hardware Local)
# Aquí está el cambio de arquitectura (Por la placa de la notebook): limitamos el context_window a 4096.Esto garantiza que el texto enviado a LM Studio no sature la memoria gráfica.
Settings.llm = OpenAILike(
    model="meta-llama-3-8b-instruct",
    api_base="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.0, # Temperatura 0.0: Para evitar alucinaciones.
    is_chat_model=True,
    context_window=4096 # Limitado para evitar cuellos de botella en la VRAM de la GPU.
)

# --- 2. CARGA DE BASE DE DATOS VECTORIAL ---
# @st.cache_resource es crucial en Streamlit. Evita que la BD se recargue en cada interacción del usuario, optimizando el rendimiento.
@st.cache_resource
def cargar_indice():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("asistente_universitario")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)

index = cargar_indice()

# --- 3. PLANTILLA ESTRICTA (Prompt Engineering, System y Role Prompting) ---
# Aquí se programa el comportamiento del asistente. Le doy un rol (Experto del DACEFyN) y le coloco una instrucción condicional (If/Else lógico): Si no está en el contexto, que no invente.
mensajes_qa = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "Eres un asistente experto del DACEFyN. Responde SIEMPRE en ESPAÑOL. "
            "Tu única fuente de verdad es el contexto proporcionado. NO inventes absolutamente nada."
        )
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "Contexto:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Pregunta: {query_str}\n"
            "Instrucción: Si la respuesta se encuentra en el contexto, extráela y responde claramente. "
            "Si la respuesta NO está en el contexto, responde ÚNICAMENTE con la frase exacta: 'No poseo la información en mis reglamentos'.\n"
            "Respuesta:"
        )
    )
]
plantilla_qa = ChatPromptTemplate(mensajes_qa)

# --- 4. MOTOR DE BÚSQUEDA ---
# similarity_top_k=7 define que solo busque los 7 fragmentos más relevantes para no saturar el contexto.(NO LLEGA A 10 o 15 en la notebook)
# response_mode="compact" obliga a LlamaIndex a hacer 1 sola consulta directa a LM Studio, evitando bucles de memoria.
query_engine = index.as_query_engine(
    similarity_top_k=7, # 7 para cuidar la memoria, 10 va a fallar y alucinar o hablar en ingles
    response_mode="compact", 
    text_qa_template=plantilla_qa
)


# --- 5. INTERFAZ DE USUARIO (Streamlit) ---
st.set_page_config(
    page_title="Asistente DACEFyN", 
    page_icon="🎓",
    layout="centered" 
)

# CSS para ocultar las marcas de agua de Streamlit y darle una presentacion de software propio.
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Panel lateral (Sidebar) con información institucional
with st.sidebar:
    st.title("⚙️ Información del Sistema")
    st.markdown("---")
    st.markdown("**Proyecto:** Asistente Virtual RAG")
    st.markdown("**Carrera:** Ingeniería en Sistemas de Información")
    st.markdown("**Departamento:** DACEFyN - UNLaR")
    st.markdown("---")
    st.info("💡 **Tip de uso:** Puedes preguntarme sobre los requisitos de inscripción, correlatividades del Plan 2024, o condiciones para mantener la regularidad.")

# Encabezado principal y Bienvenida
st.title("🎓 Asistente Administrativo")
st.markdown("#### Departamento de Ciencias Exactas, Físicas y Naturales")
st.markdown("¡Hola! Soy el asistente virtual del DACEFyN. Estoy aquí para resolver tus dudas sobre reglamentos y planes de estudio. ¿En qué te puedo ayudar hoy?")
st.markdown("---")

# Gestión del historial de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
    

# Bucle que dibuja los mensajes anteriores guardados en el historial.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# st.chat_input captura lo que el usuario escribe. El operador ':=' asigna y evalúa al mismo tiempo.
if prompt := st.chat_input("Escribe tu consulta administrativa aquí..."):
    
     # 1. Guardamos y mostramos la pregunta del usuario.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

     # 2. Mostramos el globo de carga y consultamos al motor RAG.
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los reglamentos del DACEFyN..."):
            response = query_engine.query(prompt) # El RAG trabaja aca.
            st.markdown(response.response)
            
            # 3. Un panel desplegable (expander) para la transparencia de los datos.
            # Iteramos sobre 'response.source_nodes' que contiene los fragmentos de texto recuperados de ChromaDB.
            with st.expander("📄 Ver información recuperada de la Base de Datos (Contexto RAG)"):
                for i, node in enumerate(response.source_nodes):
                    st.markdown(f"**Fragmento {i+1} (Similitud: {node.score:.2f})**")
                    st.info(node.node.text)
                    
    # 4. Guardamos la respuesta del asistente en el historial para la próxima recarga.                
    st.session_state.messages.append({"role": "assistant", "content": response.response})
    
    

    #      COMANDO PARA EJECUTAR EN VS CODE     python -m streamlit run app.py
