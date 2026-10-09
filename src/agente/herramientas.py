import base64
from io import BytesIO
from pathlib import Path

import requests
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from .voz import reconocer_audio

MODEL = "qwen3-8b"
VISION_MODEL = "qwen3-vl-8b-instruct"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


def traducir(texto: str, idioma_objetivo: str) -> str:
    """Traduce un texto utilizando el modelo local."""
    respuesta = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un traductor. Traduce el texto al idioma indicado. "
                    "Devuelve únicamente la traducción, sin explicaciones."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Idioma objetivo: {idioma_objetivo}\n"
                    f"Texto: {texto}"
                ),
            },
        ],
        temperature=0.2,
    )

    return respuesta.choices[0].message.content.strip()


def _validar_imagen(contenido: bytes, formato: str | None = None) -> str:
    """Verifica que los bytes contengan una imagen válida."""
    try:
        with Image.open(BytesIO(contenido)) as imagen:
            formato_real = imagen.format
            if formato is not None and formato_real != formato:
                raise ValueError(
                    f"Formato no válido: se esperaba {formato}, "
                    f"pero se recibió {formato_real or 'desconocido'}."
                )
            imagen.verify()

        # verify() comprueba la estructura; load() comprueba la decodificación.
        with Image.open(BytesIO(contenido)) as imagen:
            imagen.load()

    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError(
            "El contenido recibido no es una imagen válida o está dañado."
        ) from error

    if formato_real is None:
        raise ValueError("No se pudo determinar el formato de la imagen.")

    return formato_real


def descargar_imagen(url: str, ruta: str) -> None:
    """Descarga y valida una imagen antes de guardarla."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    contenido = response.content
    formato = _validar_imagen(contenido)

    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(f".{destino.name}.part")

    try:
        temporal.write_bytes(contenido)
        temporal.replace(destino)
    finally:
        temporal.unlink(missing_ok=True)

    print(f"Imagen guardada en: {destino} (formato: {formato})")


def resolver_problema_png(ruta: str) -> str:
    """Resuelve un problema matemático contenido en un PNG real."""
    imagen = Path(ruta).read_bytes()
    _validar_imagen(imagen, formato="PNG")
    imagen_base64 = base64.b64encode(imagen).decode("utf-8")

    respuesta = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Resuelve el problema matemático de esta imagen. "
                            "Explica brevemente el procedimiento y proporciona "
                            "el resultado final."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/png;base64,{imagen_base64}"
                            )
                        },
                    },
                ],
            }
        ],
    )

    return respuesta.choices[0].message.content.strip()


def transcribir_audio(ruta: str) -> str:
    """Transcribe un archivo de audio existente."""
    return reconocer_audio(Path(ruta))
