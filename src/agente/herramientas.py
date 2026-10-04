import base64
from pathlib import Path

import requests
from faster_whisper import WhisperModel
from openai import OpenAI


MODEL = "qwen3-8b"
VISION_MODEL = "qwen3-vl-8b-instruct"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


def cargar_whisper() -> WhisperModel:
    try:
        return WhisperModel(
            "small",
            device="cuda",
            compute_type="float16",
        )
    except Exception:
        print("CUDA no disponible. Usando CPU.")
        return WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
        )


whisper_model = cargar_whisper()


def traducir(texto: str, idioma_objetivo: str) -> str:
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


def descargar_imagen(url: str, ruta: str) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    Path(ruta).write_bytes(response.content)

    print(f"Imagen guardada en: {ruta}")


def resolver_problema_png(ruta: str) -> str:
    imagen = Path(ruta).read_bytes()
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
    segmentos, _ = whisper_model.transcribe(ruta)

    return " ".join(
        segmento.text.strip()
        for segmento in segmentos
    )