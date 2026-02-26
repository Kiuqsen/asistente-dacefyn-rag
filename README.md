# 🎓 Asistente Administrativo RAG - DACEFyN

Este proyecto es un asistente virtual desarrollado con Inteligencia Artificial local (RAG) para resolver dudas administrativas de ingresantes y alumnos de la carrera de Ingeniería en Sistemas de Información.

## 🚀 Tecnologías Utilizadas
* **Lenguaje:** Python
* **Interfaz Gráfica:** Streamlit
* **Orquestador RAG:** LlamaIndex
* **Extracción de Datos:** LlamaParse (Para lectura de PDFs complejos con tablas Y columnas)
* **Base de Datos Vectorial:** ChromaDB
* **Modelos (LLM y Embeddings):** Ejecutados localmente a través de LM Studio. Modelo utilizado meta-llama-3-8b-instruct


## ⚙️ Instrucciones de Instalación
1. Clonar este repositorio.
2. Instalar las dependencias ejecutando: `pip install -r requirements.txt`
3. Colocar una API Key válida de LlamaCloud en el archivo `crear_base_datos.py`.
4. Ejecutar el indexador para crear la base de datos: `python crear_base_datos.py`
5. Levantar la interfaz de usuario: `streamlit run app.py`

## 🎥 Demostración en Video
https://youtu.be/XZfxeQOdQ8k?si=TKSsf0i1Y79CqceF
