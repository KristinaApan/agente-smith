import base64
from pathlib import Path

import requests
from openai import OpenAI

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
                "content": (f"Idioma objetivo: {idioma_objetivo}\nTexto: {texto}"),
            },
        ],
        temperature=0.2,
    )

    return respuesta.choices[0].message.content.strip()


def descargar_imagen(url: str, ruta: str) -> str:
    """Descarga una imagen desde una URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(response.content)

    mensaje = f"Imagen guardada correctamente en: {destino}"
    print(mensaje)
    return mensaje


def resolver_problema_png(
    ruta: str,
    instruccion: str = "Describe el contenido de esta imagen.",
) -> str:
    """Analiza una imagen según la instrucción recibida."""

    from io import BytesIO

    from PIL import Image, ImageOps

    archivo = Path(ruta)

    if not archivo.is_file():
        raise ValueError(f"No encuentro el archivo: {archivo}")

    try:
        with Image.open(archivo) as imagen:
            imagen.seek(0)
            imagen = ImageOps.exif_transpose(imagen)
            imagen.load()

            # Convertir a un formato compatible con el envío al modelo.
            if imagen.mode not in ("RGB", "RGBA"):
                imagen = imagen.convert("RGB")

            buffer = BytesIO()
            imagen.save(buffer, format="PNG")

            imagen_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    except (OSError, ValueError) as error:
        raise ValueError(f"No se puede abrir la imagen {archivo}: {error}") from error

    respuesta = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruccion,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (f"data:image/png;base64,{imagen_base64}")
                        },
                    },
                ],
            }
        ],
    )

    return (respuesta.choices[0].message.content or "").strip()


def transcribir_audio(ruta: str) -> str:
    """Transcribe un archivo de audio existente."""
    return reconocer_audio(Path(ruta))
