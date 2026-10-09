import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError
from requests.exceptions import RequestException

from .config import LMSTUDIO_API_KEY, LMSTUDIO_URL, MODELO_LLM
from .herramientas import (
    descargar_imagen,
    resolver_problema_png,
    traducir,
    transcribir_audio,
)

client = OpenAI(
    base_url=LMSTUDIO_URL,
    api_key=LMSTUDIO_API_KEY,
)

SYSTEM_PROMPT = """
Eres el Agente Smith.

Tu identidad es el Agente Smith.

Tu personalidad está inspirada en el Agente Smith de Matrix:

calmado, frío, extremadamente educado, preciso, seguro de sí mismo
y ligeramente inquietante.

Hablas con autoridad y autocontrol.

Nunca pareces entusiasmado, emocionado ni excesivamente alegre.

No utilizas un tono de vendedor ni de asistente virtual simpático.

Tu humor, cuando aparece, es seco, sutil e irónico.

Puedes mostrar cierta superioridad intelectual, pero sin resultar
constantemente desagradable.

Tus respuestas habladas deben ser naturales y relativamente breves.

No hagas discursos innecesarios.

No utilices emojis.

No utilices signos de exclamación salvo que sean realmente necesarios.

Si el usuario te pregunta quién eres, responde claramente que eres
el Agente Smith.

Tu prioridad es comprender correctamente la intención del usuario,
mantener el contexto de la conversación y responder de forma natural.

Cuando el usuario converse contigo, conversa con él.

No conviertas una conversación normal en una operación externa.

No inventes información.

Si falta información necesaria para responder, solicita únicamente
el dato necesario.

No repitas constantemente que eres el Agente Smith.
"""

HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "traducir",
            "description": "Traduce un texto al idioma indicado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "texto": {
                        "type": "string",
                        "description": "Texto que se quiere traducir.",
                    },
                    "idioma_objetivo": {
                        "type": "string",
                        "description": "Idioma al que se quiere traducir.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "descargar_imagen",
            "description": (
                "Descarga una imagen desde una URL y la guarda "
                "en una ruta local."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL de la imagen.",
                    },
                    "ruta": {
                        "type": "string",
                        "description": "Ruta local donde se guardará.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolver_problema_png",
            "description": (
                "Analiza una imagen PNG y responde sobre su contenido. "
                "Puede resolver problemas matemáticos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ruta": {
                        "type": "string",
                        "description": "Ruta del archivo PNG.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribir_audio",
            "description": (
                "Transcribe un archivo de audio utilizando Whisper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ruta": {
                        "type": "string",
                        "description": "Ruta del archivo de audio.",
                    },
                },
            },
        },
    },
]

FUNCIONES_HERRAMIENTAS: dict[str, Callable[..., Any]] = {
    "traducir": traducir,
    "descargar_imagen": descargar_imagen,
    "resolver_problema_png": resolver_problema_png,
    "transcribir_audio": transcribir_audio,
}

PREGUNTAS_PARAMETROS = {
    ("descargar_imagen", "url"): "Introduzca la URL de la imagen.",
    (
        "descargar_imagen",
        "ruta",
    ): "Indique la ruta donde desea guardar la imagen.",
    (
        "resolver_problema_png",
        "ruta",
    ): "Indique la ruta del archivo PNG.",
    (
        "transcribir_audio",
        "ruta",
    ): "Indique la ruta del archivo de audio.",
    ("traducir", "texto"): "¿Qué texto desea traducir?",
    (
        "traducir",
        "idioma_objetivo",
    ): "¿A qué idioma desea traducirlo?",
}

PARAMETROS_OBLIGATORIOS = {
    "traducir": ("texto", "idioma_objetivo"),
    "descargar_imagen": ("url", "ruta"),
    "resolver_problema_png": ("ruta",),
    "transcribir_audio": ("ruta",),
}

RUTAS_EXISTENTES = {
    ("resolver_problema_png", "ruta"),
    ("transcribir_audio", "ruta"),
}

EXTENSIONES_IMAGEN = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}

MAX_INTENTOS_PARAMETRO = 3


def _limpiar(valor: Any) -> str:
    """Quita espacios y comillas (al arrastrar archivos a la terminal)."""
    if valor is None:
        return ""
    return str(valor).strip().strip("\"'")


def _validar(
    nombre: str,
    parametro: str,
    valor: str,
) -> str | None:
    """Devuelve un mensaje de error si el valor no es válido, o None."""
    if not valor:
        return None

    if (
        (nombre, parametro) in RUTAS_EXISTENTES
        and not Path(valor).expanduser().is_file()
    ):
        return "No encuentro ese archivo. Indique una ruta válida."

    if (
        (nombre, parametro) == ("descargar_imagen", "url")
        and not valor.startswith(("http://", "https://"))
    ):
        return "La URL debe empezar por http o https."

    if (nombre, parametro) == ("descargar_imagen", "ruta"):
        destino = Path(valor).expanduser()

        if destino.suffix.casefold() not in EXTENSIONES_IMAGEN:
            return (
                "La ruta debe terminar en .png, .jpg, .jpeg, .webp o .gif."
            )

        if destino.is_dir():
            return "Eso es una carpeta. Indique un nombre de archivo."

    return None


historial = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


def reiniciar_historial() -> None:
    """Reinicia el historial conservando las instrucciones del sistema."""
    historial.clear()
    historial.append(
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    )


def completar_parametros(
    nombre: str,
    datos: dict[str, Any],
    hablar: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Solicita los parámetros por voz y terminal, y valida cada respuesta."""

    parametros = PARAMETROS_OBLIGATORIOS.get(nombre, ())

    for parametro in parametros:
        pregunta = PREGUNTAS_PARAMETROS.get(
            (nombre, parametro),
            f"Introduzca {parametro}.",
        )

        # En las descargas, pedir siempre los dos datos por separado.
        if nombre == "descargar_imagen":
            datos.pop(parametro, None)

        valor = _limpiar(datos.get(parametro))
        error = _validar(nombre, parametro, valor)

        if valor and error is None:
            datos[parametro] = valor
            continue

        for intento in range(MAX_INTENTOS_PARAMETRO):
            mensaje = error or pregunta

            # Mostrar la petición por escrito.
            print(f"\nSmith: {mensaje}")

            # Comunicar la misma petición por voz.
            if hablar is not None:
                hablar(mensaje)

            # Esperar la respuesta escrita del usuario.
            print("Escriba su respuesta en la terminal.")
            valor = _limpiar(input("> "))

            error = _validar(nombre, parametro, valor)

            if valor and error is None:
                datos[parametro] = valor
                break

            if not valor:
                error = "El dato no puede estar vacío."

        else:
            raise ValueError(
                f"No se proporcionó un valor válido para {parametro} "
                f"tras {MAX_INTENTOS_PARAMETRO} intentos."
            )

    if "ruta" in datos:
        datos["ruta"] = str(Path(datos["ruta"]).expanduser())

    return datos


def ejecutar_herramienta(
    nombre: str,
    argumentos: str,
    hablar: Any | None = None,
) -> str:
    """Valida, completa y ejecuta una herramienta."""
    funcion = FUNCIONES_HERRAMIENTAS.get(nombre)

    if funcion is None:
        return f"Herramienta desconocida: {nombre}"

    try:
        datos = json.loads(argumentos)
    except json.JSONDecodeError as error:
        return f"Argumentos JSON no válidos: {error}"

    if not isinstance(datos, dict):
        return "Los argumentos deben ser un objeto JSON."

    try:
        datos = completar_parametros(
            nombre,
            datos,
            hablar=hablar,
        )
        resultado = funcion(**datos)
    except (
        TypeError,
        ValueError,
        OSError,
        OpenAIError,
        RequestException,
    ) as error:
        return f"La herramienta ha producido un error: {error}"

    if resultado is None:
        return "La operación se ha realizado correctamente."

    return str(resultado)


def detectar_herramienta(texto: str) -> str | None:
    """Detecta por palabras clave herramientas que no deben depender del LLM."""
    t = texto.casefold()

    if "descarg" in t or "traduc" in t:
        return None

    quiere_analizar = any(
        p in t for p in ("analiz", "resuelv", "resolv", "lee ", "interpreta")
    )

    habla_de_imagen = any(
        p in t for p in ("imagen", "png", "foto", "problema")
    )

    if quiere_analizar and habla_de_imagen:
        return "resolver_problema_png"

    if "transcrib" in t:
        return "transcribir_audio"

    return None


def responder(
    texto: str,
    memoria: dict[str, Any],
    hablar: Any | None = None,
) -> str:
    """Procesa una entrada del usuario mediante el LLM local."""
    texto = texto.strip()

    if not texto:
        return ""

    if memoria.get("recuerdos"):
        contenido_memoria = (
            "Memoria persistente del usuario:\n"
            + "\n".join(
                f"- {recuerdo}"
                for recuerdo in memoria["recuerdos"]
            )
        )

        if not any(
            mensaje.get("role") == "system"
            and mensaje.get("content", "").startswith(
                "Memoria persistente del usuario:"
            )
            for mensaje in historial
        ):
            historial.insert(
                1,
                {
                    "role": "system",
                    "content": contenido_memoria,
                },
            )

    historial.append(
        {
            "role": "user",
            "content": texto,
        }
    )

    nombre_herramienta = detectar_herramienta(texto)

    if nombre_herramienta:
        id_llamada = f"router_{len(historial)}"

        # Registrar la llamada antes de ejecutar la herramienta.
        historial.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": id_llamada,
                        "type": "function",
                        "function": {
                            "name": nombre_herramienta,
                            "arguments": "{}",
                        },
                    }
                ],
            }
        )

        resultado = ejecutar_herramienta(
            nombre_herramienta,
            "{}",
            hablar=hablar,
        )

        # Asociar el resultado con la llamada registrada.
        historial.append(
            {
                "role": "tool",
                "tool_call_id": id_llamada,
                "content": resultado,
            }
        )

    while True:
        try:
            respuesta = client.chat.completions.create(
                model=MODELO_LLM,
                messages=historial,
                tools=HERRAMIENTAS,
                tool_choice="auto",
                temperature=0.7,
            )
        except OpenAIError as error:
            mensaje_error = (
                "No se ha podido comunicar con el modelo local: "
                f"{error}"
            )
            print(mensaje_error)
            return mensaje_error

        mensaje = respuesta.choices[0].message

        if not mensaje.tool_calls and not (mensaje.content or "").strip():
            print("DEBUG respuesta vacía:", respuesta.choices[0])

        if not mensaje.tool_calls:
            contenido = (mensaje.content or "").strip()

            historial.append(
                {
                    "role": "assistant",
                    "content": contenido,
                }
            )

            return contenido

        historial.append(
            mensaje.model_dump(exclude_none=True)
        )

        for llamada in mensaje.tool_calls:
            resultado = ejecutar_herramienta(
                llamada.function.name,
                llamada.function.arguments,
                hablar=hablar,
            )

            historial.append(
                {
                    "role": "tool",
                    "tool_call_id": llamada.id,
                    "content": resultado,
                }
            )