# Asistente IA — Ejercicio 3.2.2

## Descripción

En este ejercicio he desarrollado un asistente en Python con un menú interactivo que permite realizar diferentes tareas relacionadas con inteligencia artificial.

Para las funciones que necesitan un modelo de IA he utilizado **LM Studio**, trabajando con modelos de forma local y sin utilizar APIs de pago.

## Funcionalidades

1. Traducir texto
   Recibe un texto y el idioma de destino mediante un objeto JSON y realiza la traducción utilizando un modelo Qwen ejecutado desde LM Studio.

2. Descargar una imagen
   Permite introducir la URL de una imagen y guardarla en el sistema local.

3. Resolver un problema desde una imagen PNG
   Permite introducir la ruta de una imagen PNG que contiene un problema. El modelo **Qwen3-VL** analiza la imagen, explica brevemente el procedimiento y proporciona el resultado.

4. Transcribir audio
   Permite introducir la ruta de un archivo de audio y obtener su transcripción utilizando **faster-whisper**.

5. Salir
   Finaliza el programa.

## Tecnologías utilizadas

1. Python
2. uv
3. LM Studio
4. Qwen3 8B
5. Qwen3-VL 8B Instruct
6. OpenAI Python client
7. Requests
8. faster-whisper

## Requisitos

1. Python
2. uv
3. LM Studio
4. Los modelos utilizados por el proyecto cargados en LM Studio

## Instalación

Una vez descargado el proyecto, las dependencias se pueden instalar con:

```bash
uv sync
```

## Configuración de LM Studio

Para las funciones que utilizan inteligencia artificial, LM Studio debe estar ejecutándose con su servidor local disponible en:

```text
http://localhost:1234/v1
```

El proyecto utiliza el cliente de Python compatible con la API de OpenAI, pero las peticiones se realizan al servidor local de LM Studio.

## Ejecución

Para iniciar el asistente:

```bash
uv run main.py
```

Después aparecerá el menú interactivo para seleccionar la función que se quiera utilizar.

## Archivos de prueba

1. `problema.png` — imagen utilizada para probar la resolución de problemas.
2. `Spanish_Can_I_sit_here.wav` — archivo de audio utilizado para probar la transcripción.

## Objetivo del ejercicio

El objetivo ha sido integrar diferentes funcionalidades relacionadas con IA dentro de un mismo programa Python, organizando cada tarea en una función independiente y controlando su ejecución mediante un menú interactivo.
