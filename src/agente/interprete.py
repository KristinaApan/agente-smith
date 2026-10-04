import json

from openai import OpenAI


MODEL = "qwen3-8b"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


SYSTEM_PROMPT = """
Eres el intérprete de órdenes de un asistente de voz.

Tu tarea es identificar qué acción quiere realizar el usuario.

Las acciones disponibles son:

- traducir
- descargar_imagen
- resolver_imagen
- transcribir_audio

Debes responder ÚNICAMENTE con JSON válido.
Nunca utilices Markdown.
Nunca añadas explicaciones.

Formato:

{
    "accion": "...",
    "parametros": {
        "...": "..."
    }
}

Para los parámetros que el usuario todavía no haya proporcionado,
utiliza null.

Parámetros:

traducir:
    texto
    idioma_objetivo

descargar_imagen:
    url

resolver_imagen:
    ruta_png

transcribir_audio:
    ruta_audio
"""


def interpretar(texto: str) -> dict:
    respuesta = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": texto},
        ],
        temperature=0,
    )

    contenido = respuesta.choices[0].message.content.strip()

    return json.loads(contenido)


if __name__ == "__main__":
    resultado = interpretar("Asistente, descarga la imagen.")
    print(resultado)
