# Agente Smith

## Descripción

Agente Smith es un asistente personal de inteligencia artificial desarrollado en Python que reúne conversación inteligente, interacción por voz, memoria persistente y una interfaz gráfica propia. Está diseñado para interpretar instrucciones, utilizar herramientas y facilitar tareas desde el ordenador, con especial atención al procesamiento local y la privacidad.

## Arquitectura y tecnologías

**Python** es la base del proyecto, cuya estructura modular permite desarrollar y mantener sus componentes de forma independiente.

**LM Studio** proporciona el entorno para ejecutar modelos de lenguaje localmente. Smith utiliza **Qwen3-8B** para interpretar peticiones, generar respuestas y decidir cuándo recurrir a sus herramientas. **Qwen3-VL-8B Instruct** está destinado al análisis de imágenes y a la ampliación de sus capacidades visuales.

**Faster-Whisper** se encarga del reconocimiento del habla, mientras que **Qwen3-TTS** genera las respuestas habladas a partir de una referencia de voz personalizada.

**PyTorch y CUDA** permiten acelerar mediante la GPU NVIDIA los componentes compatibles. **JSON** facilita la gestión de datos y la memoria persistente, mientras que **uv** y **Git** ayudan a mantener un entorno de desarrollo organizado y un control de versiones adecuado.

## Funcionalidades

**Conversación inteligente.** Interpreta instrucciones mediante un modelo de lenguaje local y puede recurrir a herramientas para realizar tareas concretas.

**Interfaz gráfica propia.** Permite interactuar con Smith desde un entorno visual diseñado para facilitar su uso.

**Interacción por voz.** Reconoce instrucciones habladas y genera respuestas sonoras utilizando una voz personalizada.

**Memoria persistente.** Conserva información relevante entre sesiones para mantener el contexto de las conversaciones.

**Traducción de textos.** Traduce contenido al idioma solicitado mediante una herramienta basada en datos estructurados en JSON.

**Gestión de imágenes y audio.** Permite descargar imágenes desde una URL y transcribir archivos de audio a texto.

**Análisis visual.** El proyecto contempla el uso de modelos multimodales para interpretar imágenes y abordar problemas a partir de su contenido.

## Tecnologías utilizadas

* Python
* uv
* LM Studio
* Qwen3-8B
* Qwen3-VL-8B Instruct
* OpenAI Python Client
* Faster-Whisper
* Qwen3-TTS
* PyTorch y CUDA
* JSON
* Git

## Requisitos

* Python en una versión compatible con el proyecto.
* uv para gestionar el entorno virtual y las dependencias.
* LM Studio para ejecutar los modelos de lenguaje localmente.
* Los modelos necesarios descargados y disponibles en LM Studio.
* Una GPU NVIDIA compatible con CUDA para acelerar los componentes que lo admiten.

## Instalación

Clona el repositorio y accede a la carpeta del proyecto:

```bash
git clone https://github.com/KristinaApan/agente-smith.git
cd agente-smith
```

Instala las dependencias:

```bash
uv sync
```

## Ejecución

Inicia el asistente desde la raíz del proyecto:

```bash
uv run main.py
```


Instala las dependencias definidas en el proyecto:

```bash
uv sync
```

`uv sync` crea o actualiza el entorno virtual y sincroniza las dependencias declaradas en `pyproject.toml` con el archivo de bloqueo del proyecto, si está disponible.

## Configuración de LM Studio

Inicia LM Studio, descarga los modelos necesarios y arranca el servidor local compatible con la API de OpenAI.

La dirección utilizada por el proyecto es:

```text
http://localhost:1234/v1
```

El cliente de Python se comunica con este servidor local, por lo que las peticiones a los modelos configurados no necesitan utilizar una API de pago externa.

## Ejecución

Desde la carpeta raíz del proyecto, inicia Smith con:

```bash
uv run main.py
```

Si el proyecto utiliza un archivo de entrada diferente, ejecuta el punto de entrada correspondiente.

## Un proyecto en evolución

Agente Smith continúa creciendo con una arquitectura modular que facilita la incorporación de nuevas herramientas y modelos. El desarrollo se centra en mejorar la integración entre sus componentes, perfeccionar la interfaz y ampliar sus capacidades de interacción y asistencia.

La meta es construir un asistente personal cada vez más completo y adaptable, capaz de comprender instrucciones, conservar información útil y ayudar con tareas desde un entorno propio.

**Smith tiene una base sobre la que seguir construyendo: lo que puede hacer hoy es solo el comienzo de lo que podrá llegar a hacer.**
