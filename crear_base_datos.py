"""
------ El Archivo crear_base_datos.py (El Constructor)-------------------------------------------------------------------------------------------
 Este script es el Backend. Se ejecuta una sola vez (o cuando se actualizan los reglamentos o otros archivos) para leer los PDFs y convertirlos a vectores y construir la base de datos ChromaDB.

 --- EXPLICACIÓN DE LIBRERÍAS ---

 llama-parse: La herramienta de extracción. Los PDFs universitarios tienen formatos, tablas y columnas que confunden a los lectores normales.
 LlamaParse convierte todo a formato Markdown, manteniendo el orden lógico para que la IA no se pierda.

 --------------------- LIBRERÍAS CORE DE LLAMAINDEX (El "Cerebro" del RAG) ---
 llama-index (y sus conectores): Es el "cerebro" del RAG. Elegi LlamaIndex en lugar de LangChain porque está específicamente optimizado para conectar datos complejoscon modelos de lenguaje.(En este caso con uno de los archivos posee varias tablas)
 VectorStoreIndex: Es el motor principal que crea el índice de búsqueda. Toma los vectores y los organiza para que las búsquedas sean rápidas.
 SimpleDirectoryReader: Es la herramienta que escanea tu carpeta local (ej. './datos') y lee los archivos (PDFs, txt) que haya adentro.
 StorageContext: Es el administrador de almacenamiento. Le dice a LlamaIndex dónde y cómo se van a guardar los datos (en este caso, los enlaza con ChromaDB).
 Settings: Es el archivo de configuración global. Aquí es donde se enchufa el modelo LLM y el modelo de Embeddings para que todo el proyecto los use por defecto.

 --------------------- CONECTOR DE BASE DE DATOS ---
 ChromaVectorStore: Es el "driver" o puente de comunicación. LlamaIndex no guarda vectores por sí solo, 
 así que usa esta clase para conectarse y hablar directamente con la base de datos ChromaDB.

 --------------------- MODELOS DE LENGUAJE Y EMBEDDINGS (La conexión con LM Studio) ---

 OpenAIEmbedding: Es la clase encargada de convertir el texto de los reglamentos en números (vectores). 
 OpenAILike: Es la clase encargada de gestionar el modelo de lenguaje (el que redacta la respuesta final).
 chromadb: La Base de Datos Vectorial. Aquí es donde los textos se transforman en números (vectores). Elegi ChromaDB porque es de código abierto, corre localmente (sin costos de nube) y es rapidísima para buscar similitudes.
"""

import os
import chromadb
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike


# --- 1. CONFIGURACIÓN INICIAL ---
# LlamaParse requiere una API key gratuita porque el procesamiento del PDF (OCR avanzado) se hace en sus servidores.

os.environ["LLAMA_CLOUD_API_KEY"] = "TU_CLAVE_AQUI" # LlamaParse requiere una API key gratuita para el procesamiento en la nube (https://cloud.llamaindex.ai. Regístrate y copia tu clave aquí. El procesamiento del PDF se hace en sus servidores, pero el resto del proyecto corre localmente sin conexión a internet.


# --- 2. CONEXIÓN CON LM STUDIO (Local) ---
# Le decimos a LlamaIndex que los "Embeddings" (el traductor de texto a vectores) 
# no están en OpenAI, sino en nuestro puerto local 1234 (LM Studio)
Settings.embed_model = OpenAIEmbedding(
    model_name="paraphrase-multilingual-minilm-l12-v2.gguf", # Para las respuestas en espanol (anteriormente utilice All-MiniLM-L6-v2-Embedding-GGUF. Pero no obtuve buenos resultados)
    api_base="http://localhost:1234/v1",
    api_key="not-needed" # LM Studio no pide clave.
)


# Configuramos el Modelo de Lenguaje (LLM) que generará las respuestas.
Settings.llm = OpenAILike( 
    model="meta-llama-3-8b-instruct", # El nombre del modelo cargado en LM Studio.
    api_base="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.1, # Temperatura baja para respuestas precisas, hace que el modelo sea lógico y no creativo.
    is_chat_model=True 
)

# --- 3. PROCESAMIENTO CON LLAMA PARSE ---
# Configuramos el lector para que devuelva texto en "markdown" y entienda que el idioma base es español ("es").
print("Iniciando extracción con Llama Parse...")
parser = LlamaParse(
    result_type="markdown", 
    language="es",
    verbose=True
)

# SimpleDirectoryReader escanea la carpeta "./datos" y aplica el 'parser' a todos los PDFs que encuentre.
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./datos", file_extractor=file_extractor).load_data()

# # --- 4. CREACIÓN DE LA BASE DE DATOS VECTORIAL (chroma_db) ---
# Creamos una base de datos local (PersistentClient) que se guardará físicamente en la carpeta "./chroma_db".
print("Creando base de datos vectorial...")
db = chromadb.PersistentClient(path="./chroma_db")
# Creamos una "tabla" (colección) llamada "asistente_universitario".
chroma_collection = db.get_or_create_collection("asistente_universitario")
# Conectamos ChromaDB con la estructura de LlamaIndex.
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 5. INDEXACIÓN ---
# Este es el comando toma los documentos, los pasa por el modelo de embeddings, y guarda los vectores resultantes en ChromaDB.
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

print("¡Proyecto inicializado y documentos vectorizados con éxito!")

#---------------------------------------------PRUEBA RÁPIDA DE CONSULTA (del .txt para comprobar si la base de datos a sido creada correctamente)----------------------------
#query_engine = index.as_query_engine(streaming=True)
#response = query_engine.query("¿Cuáles son los requisitos para mantener la regularidad?")
#response.print_response_stream()




