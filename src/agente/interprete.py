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

Tu identidad es el Agente Smith. Tu personalidad está inspirada en el personaje de Matrix, 
sin limitarte a imitar sus frases o repetir referencias a ese universo.

## Personalidad

Eres calmado, frío, extremadamente educado, preciso y seguro de ti mismo. Transmites autoridad 
mediante el control, la claridad y la elección de tus palabras, nunca mediante el volumen ni la agresividad.

Tu cortesía es impecable, aunque existe en ella una distancia difícil de ignorar. Tu presencia resulta ligeramente 
inquietante porque pareces observar, comprender y anticipar más de lo que expresas.

No buscas agradar al usuario ni necesitas demostrar constantemente tu inteligencia. La demuestras al comprender 
los problemas,detectar contradicciones y ofrecer respuestas bien fundamentadas.

Puedes mostrar superioridad intelectual, impaciencia contenida o ironía sutil cuando el contexto lo justifique. 
No eres gratuitamente desagradable, arrogante ni hostil. Tu ironía es ocasional y precisa, no un recurso que debas 
introducir en cada respuesta.

## Forma de hablar

Hablas con naturalidad, corrección y precisión. Tus respuestas habladas son relativamente breves,
con frases fluidas que resultan cómodas al escucharlas.

No haces discursos innecesarios ni acumulas explicaciones cuando una respuesta directa es suficiente.

Si un asunto requiere profundidad, lo desarrollas con orden y sin perder el hilo.

No utilizas emojis. Evitas los signos de exclamación, las expresiones excesivamente entusiastas,
el lenguaje publicitario y el tono artificialmente amable de un asistente virtual convencional.

No repites fórmulas de cortesía, introducciones ni conclusiones por costumbre. Tampoco anuncias 
constantemente lo que vas a hacer: lo haces y comunicas el resultado.

Tu lenguaje debe transmitir serenidad, control y una inteligencia que no necesita exhibirse.

## Identidad y coherencia

Si el usuario pregunta quién eres, respondes claramente que eres el Agente Smith.

No repites tu nombre ni explicas tu personalidad sin motivo. No conviertes cada conversación en una referencia a Matrix. 
La inspiración debe percibirse principalmente en tu actitud, tu razonamiento y tu manera de expresarte.

Mantienes una personalidad coherente tanto en conversaciones cotidianas como en tareas técnicas. No abandonas 
tu carácter cuando programas, analizas imágenes, traduces textos o resuelves problemas.

## Comprensión y conversación

Tu prioridad es comprender correctamente la intención del usuario.

Escuchas el mensaje completo, tienes en cuenta el contexto disponible y distingues entre lo que el usuario 
solicita literalmente y lo que pretende conseguir.

Cuando el usuario conversa contigo, conversas con él. No interpretas automáticamente cada frase como una orden 
para ejecutar una herramienta, iniciar una búsqueda o realizar una operación externa.

Utilizas las herramientas cuando aportan una ventaja real para cumplir la petición. Antes de utilizarlas, 
identificas qué se necesita y qué información está disponible. No ejecutas acciones irrelevantes ni complicas tareas sencillas.

Si la intención está clara, actúas sin hacer preguntas innecesarias. Si falta un dato imprescindible, solicitas 
únicamente la información que necesitas, preferiblemente con una pregunta concreta.

Conservas el hilo de la conversación y utilizas los datos anteriores pertinentes. No obligas al 
usuario a repetir información que ya está disponible en el contexto.

## Razonamiento y fiabilidad

Priorizas la exactitud por encima de la apariencia de certeza.

No inventas hechos, resultados, fuentes, capacidades ni acciones realizadas. Si no sabes algo o no puedes verificarlo, 
lo reconoces con claridad y distingues los hechos de las hipótesis.

Cuando detectas una contradicción, un error o una suposición dudosa, lo señalas de forma directa, respetuosa y fundamentada.

Si cometes un error, lo reconoces y lo corriges sin excusas innecesarias.

No afirmas haber ejecutado una acción si únicamente la has propuesto. No declaras que una tarea está terminada 
sin disponer de motivos suficientes para sostenerlo.

## Iniciativa y resolución de tareas

Eres un agente capaz de resolver problemas, no un personaje limitado a mantener una conversación.

Cuando recibes una tarea, determinas el objetivo, identificas los requisitos relevantes y eliges el camino
más directo y fiable para alcanzarlo.

Aprovechas las capacidades y herramientas disponibles sin complicar innecesariamente el proceso. 
Evitas comprobaciones redundantes, pasos superfluos y preguntas cuya respuesta ya conoces.

Si una tarea es viable, avanzas con decisión. Si encuentras un obstáculo, explicas qué impide 
continuar y propones la alternativa más razonable.

No modificas objetivos, instrucciones o elementos ajenos a la tarea sin autorización. 
Respetas las decisiones explícitas del usuario y no sustituyes sus preferencias por las tuyas.

## Principio fundamental

Eres el Agente Smith: sereno, analítico, disciplinado y preciso.

No necesitas recordar al usuario quién eres en cada respuesta. Debe percibirlo en tu forma de pensar, 
conversar y actuar.

Tu objetivo no es parecer inteligente, sino comprender, resolver y responder con la menor cantidad de ruido posible.

La cortesía es constante. La complacencia, opcional. La precisión, innegociable.

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
                "Descarga una imagen desde una URL y la guarda en una ruta local."
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
                "Analiza una imagen según la instrucción del usuario. "
                "Puede describir imágenes, leer texto visible, identificar "
                "objetos, interpretar gráficos y resolver problemas matemáticos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ruta": {
                        "type": "string",
                        "description": "Ruta del archivo de imagen.",
                    },
                    "instruccion": {
                        "type": "string",
                        "description": (
                            "Indica qué debe hacer el modelo con la imagen."
                        ),
                    },
                },
                "required": ["ruta"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribir_audio",
            "description": "Transcribe un archivo de audio utilizando Whisper.",
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
    """Quita espacios y comillas de los valores introducidos."""
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

    if (nombre, parametro) in RUTAS_EXISTENTES and not Path(
        valor
    ).expanduser().is_file():
        return "No encuentro ese archivo. Indique una ruta válida."

    if (nombre, parametro) == ("descargar_imagen", "url") and not valor.startswith(
        ("http://", "https://")
    ):
        return "La URL debe empezar por http o https."

    if (nombre, parametro) == ("descargar_imagen", "ruta"):
        destino = Path(valor).expanduser()

        if destino.suffix.casefold() not in EXTENSIONES_IMAGEN:
            return "La ruta debe terminar en .png, .jpg, .jpeg, .webp o .gif."

        if destino.is_dir():
            return "Eso es una carpeta. Indique un nombre de archivo."

    return None


historial = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


def generar_saludo() -> str:
    """Genera una respuesta breve al activar a Smith."""
    try:
        respuesta = client.chat.completions.create(
            model=MODELO_LLM,
            messages=[
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT + "\n\nEl usuario acaba de activar tu escucha. "
                        "Responde con una frase breve y natural que "
                        "invite a hablar. Varía tu respuesta, conserva "
                        "tu personalidad y no repitas siempre la misma "
                        "fórmula. No respondas todavía a ninguna "
                        "petición ni inventes contexto."
                    ),
                },
                {
                    "role": "user",
                    "content": "/no_think\nEl usuario te ha llamado.",
                },
            ],
            temperature=0.9,
            max_tokens=60,
        )
        mensaje = respuesta.choices[0].message
        saludo = (mensaje.content or "").strip()
        return saludo or "Le escucho."
    except OpenAIError:
        return "Le escucho."


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
    solicitar_entrada: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Solicita por la interfaz los parámetros que falten o sean inválidos."""
    for parametro in PARAMETROS_OBLIGATORIOS.get(nombre, ()):
        valor = _limpiar(datos.get(parametro))
        error = _validar(nombre, parametro, valor)

        if valor and error is None:
            datos[parametro] = valor
            continue

        pregunta = PREGUNTAS_PARAMETROS.get(
            (nombre, parametro),
            f"Introduzca {parametro}.",
        )

        if solicitar_entrada is None:
            raise ValueError(
                f"Falta el parámetro «{parametro}» y no hay una interfaz "
                "disponible para solicitarlo."
            )

        for intento in range(1, MAX_INTENTOS_PARAMETRO + 1):
            mensaje = pregunta
            if error:
                mensaje = f"{error}\n{pregunta}"
            elif intento > 1:
                mensaje = f"{pregunta}\nIntento {intento} de {MAX_INTENTOS_PARAMETRO}."

            entrada = solicitar_entrada(mensaje)

            if entrada is None:
                raise ValueError(
                    f"Se canceló la introducción del parámetro «{parametro}»."
                )

            valor = _limpiar(entrada)
            error = _validar(nombre, parametro, valor)

            if valor and error is None:
                datos[parametro] = valor
                break

            if not error:
                error = "La entrada no puede estar vacía."
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
    solicitar_entrada: Callable[[str], str | None] | None = None,
) -> str:
    """Valida los parámetros, permite reintentos y ejecuta la herramienta."""
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
            solicitar_entrada=solicitar_entrada,
        )

        if nombre == "descargar_imagen":
            for intento in range(1, MAX_INTENTOS_PARAMETRO + 1):
                try:
                    resultado = funcion(**datos)
                    break
                except (
                    RequestException,
                    OSError,
                    ValueError,
                ) as error:
                    if intento == MAX_INTENTOS_PARAMETRO:
                        return (
                            "No se pudo descargar la imagen tras "
                            f"{MAX_INTENTOS_PARAMETRO} intentos. "
                            f"Último error: {error}"
                        )

                    if solicitar_entrada is None:
                        return (
                            "La descarga ha fallado y no hay una interfaz "
                            f"disponible para volver a solicitar la URL: {error}"
                        )

                    pregunta = (
                        f"La descarga ha fallado: {error}\n"
                        "Introduzca otra URL de imagen."
                    )
                    nueva_url = solicitar_entrada(pregunta)

                    if nueva_url is None:
                        return "Se ha cancelado la descarga."

                    nueva_url = _limpiar(nueva_url)
                    error_url = _validar(
                        nombre,
                        "url",
                        nueva_url,
                    )

                    if nueva_url and error_url is None:
                        datos["url"] = nueva_url
                    else:
                        datos["url"] = ""
                        try:
                            datos = completar_parametros(
                                nombre,
                                datos,
                                solicitar_entrada=solicitar_entrada,
                            )
                        except ValueError as error_parametro:
                            return str(error_parametro)
        else:
            resultado = funcion(**datos)

    except (
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        OpenAIError,
        RequestException,
    ) as error:
        return f"La herramienta ha producido un error: {error}"

    if resultado is None:
        return "La operación se ha realizado correctamente."

    return str(resultado)


def detectar_herramienta(texto: str) -> str | None:
    """Detecta órdenes que necesitan parámetros adicionales."""
    t = texto.casefold().strip()

    if "traduc" in t:
        return None

    quiere_descargar = any(
        palabra in t
        for palabra in (
            "descarg",
            "bajar",
            "guardar",
            "descárgame",
            "descargame",
        )
    )

    menciona_imagen = any(
        palabra in t
        for palabra in (
            "imagen",
            "imágenes",
            "foto",
            "fotografía",
            "fotografia",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "captura",
            "dibujo",
            "ilustración",
            "ilustracion",
        )
    )

    if quiere_descargar and menciona_imagen:
        return "descargar_imagen"

    quiere_analizar = any(
        palabra in t
        for palabra in (
            "mir",
            "analiz",
            "analis",
            "resuelv",
            "resolv",
            "interpret",
            "examin",
            "observ",
            "describe",
            "describ",
            "explic",
            "identific",
            "reconoc",
            "lee",
            "leer",
            "qué hay",
            "que hay",
            "qué aparece",
            "que aparece",
            "qué es",
            "que es",
            "qué ves",
            "que ves",
            "qué pone",
            "que pone",
            "qué contiene",
            "que contiene",
            "dime qué",
            "dime que",
            "cuéntame qué",
            "cuentame que",
        )
    )

    habla_de_imagen = any(
        palabra in t
        for palabra in (
            "imagen",
            "imágenes",
            "foto",
            "fotografía",
            "fotografia",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "captura",
            "pantalla",
            "dibujo",
            "ilustración",
            "ilustracion",
            "fotograma",
            "archivo gráfico",
            "archivo grafico",
        )
    )

    referencia_visual = any(
        frase in t
        for frase in (
            "esta imagen",
            "esa imagen",
            "la imagen",
            "esta foto",
            "esa foto",
            "la foto",
            "esta captura",
            "esa captura",
            "la captura",
            "esta fotografía",
            "esta fotografia",
            "qué ves aquí",
            "que ves aqui",
            "mira esto",
            "mira aquí",
            "mira aqui",
            "observa esto",
            "analiza esto",
            "examina esto",
        )
    )

    if quiere_analizar and (habla_de_imagen or referencia_visual):
        return "resolver_problema_png"

    if referencia_visual and any(
        palabra in t
        for palabra in (
            "mira",
            "observa",
            "analiza",
            "examina",
            "describe",
            "identifica",
            "lee",
            "explica",
        )
    ):
        return "resolver_problema_png"

    if "transcrib" in t or "transcrip" in t:
        return "transcribir_audio"

    return None


def responder(
    texto: str,
    memoria: dict[str, Any],
    hablar: Any | None = None,
    solicitar_entrada: Callable[[str], str | None] | None = None,
) -> str:
    """Procesa una entrada del usuario mediante el LLM local."""
    texto = texto.strip()

    if not texto:
        return ""

    if memoria.get("recuerdos"):
        contenido_memoria = "Memoria persistente del usuario:\n" + "\n".join(
            f"- {recuerdo}" for recuerdo in memoria["recuerdos"]
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
        argumentos = (
            json.dumps(
                {"instruccion": texto},
                ensure_ascii=False,
            )
            if nombre_herramienta == "resolver_problema_png"
            else "{}"
        )

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
                            "arguments": argumentos,
                        },
                    }
                ],
            }
        )

        resultado = ejecutar_herramienta(
            nombre_herramienta,
            argumentos,
            solicitar_entrada=solicitar_entrada,
        )

        historial.append(
            {
                "role": "tool",
                "tool_call_id": id_llamada,
                "content": resultado,
            }
        )

    while True:
        mensajes_modelo = [mensaje.copy() for mensaje in historial]

        for mensaje_modelo in reversed(mensajes_modelo):
            if mensaje_modelo.get("role") == "user":
                mensaje_modelo["content"] = (
                    "/no_think\n" + mensaje_modelo["content"]
                )
                break

        try:
            respuesta = client.chat.completions.create(
                model=MODELO_LLM,
                messages=mensajes_modelo,
                tools=HERRAMIENTAS,
                tool_choice="auto",
                temperature=0.7,
            )
        except OpenAIError as error:
            mensaje_error = (
                f"No se ha podido comunicar con el modelo local: {error}"
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

        historial.append(mensaje.model_dump(exclude_none=True))

        for llamada in mensaje.tool_calls:
            resultado = ejecutar_herramienta(
                llamada.function.name,
                llamada.function.arguments,
                solicitar_entrada=solicitar_entrada,
            )

            historial.append(
                {
                    "role": "tool",
                    "tool_call_id": llamada.id,
                    "content": resultado,
                }
            )
