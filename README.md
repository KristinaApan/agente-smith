# Agente Smith

Agente Smith es un asistente personal de inteligencia artificial desarrollado en Python que combina conversación inteligente, interacción por voz, memoria persistente y una interfaz gráfica propia. Está diseñado para interpretar instrucciones, utilizar herramientas y facilitar tareas desde el ordenador, con especial atención al procesamiento local y la privacidad.

Python constituye la base del proyecto, cuya arquitectura modular permite desarrollar y mantener sus diferentes componentes de forma independiente. LM Studio proporciona el entorno para ejecutar modelos de lenguaje localmente. Smith utiliza Qwen3-8B para interpretar peticiones, generar respuestas y decidir cuándo recurrir a sus herramientas, mientras que Qwen3-VL-8B Instruct permite trabajar con capacidades de análisis visual.

Faster-Whisper se encarga del reconocimiento del habla y Qwen3-TTS genera respuestas habladas utilizando una referencia de voz personalizada. PyTorch y CUDA permiten aprovechar la GPU NVIDIA para acelerar los componentes compatibles. JSON se utiliza para gestionar datos y conservar la memoria entre sesiones, mientras que uv facilita la gestión del entorno y las dependencias y Git permite controlar las versiones del proyecto.

Smith dispone de una interfaz gráfica propia desde la que se puede interactuar con el asistente mediante texto y voz. Puede mantener conversaciones, interpretar instrucciones, utilizar herramientas para realizar tareas concretas, traducir textos, descargar imágenes desde una URL y transcribir archivos de audio. También incorpora memoria persistente para conservar información relevante entre sesiones y contempla el análisis de imágenes mediante modelos multimodales.

El proyecto está desarrollado con Python y utiliza tecnologías como LM Studio, Qwen3-8B, Qwen3-VL-8B Instruct, OpenAI Python Client, Faster-Whisper, Qwen3-TTS, PyTorch, CUDA, JSON, uv y Git.

Para ejecutar Agente Smith es necesario disponer de una versión compatible de Python, instalar uv, sincronizar las dependencias del proyecto y configurar LM Studio con los modelos necesarios. La comunicación con los modelos se realiza a través del servidor local compatible con la API de OpenAI, disponible en `http://localhost:1234/v1`, lo que permite trabajar con los modelos configurados sin depender de una API de pago externa.

Para instalar el proyecto, clona el repositorio desde GitHub y accede a su directorio con los siguientes comandos:

```bash
git clone https://github.com/KristinaApan/agente-smith.git
cd agente-smith
uv sync
```

Una vez instaladas las dependencias y configurado LM Studio, inicia el asistente desde la carpeta raíz del proyecto:

```bash
uv run main.py
```

Agente Smith es un proyecto en evolución cuya arquitectura modular permite incorporar nuevas herramientas, integrar modelos y ampliar sus capacidades progresivamente. El desarrollo se centra en mejorar la comunicación entre sus componentes, perfeccionar la interfaz gráfica y seguir ampliando sus posibilidades de interacción y asistencia.

La meta es construir un asistente personal cada vez más completo y adaptable, capaz de comprender instrucciones, conservar información útil y ayudar con distintas tareas desde su propio entorno. Smith cuenta con una base funcional sobre la que seguir creciendo, y sus posibilidades futuras dependerán de las nuevas herramientas y capacidades que se integren en el proyecto.
