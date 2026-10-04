import subprocess
import tempfile
from openai import OpenAI
from .interprete import interpretar
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import pyttsx3
import sounddevice as sd
import soundfile as sf


MODEL = "qwen3-8b"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


CUDA_DLL_DIRECTORIES = [] 


def configurar_cuda_dll() -> None:
    """Registra las DLL de CUDA/cuDNN del entorno virtual."""
    spec = importlib.util.find_spec("nvidia")

    if spec is None or spec.submodule_search_locations is None:
        return

    raiz_nvidia = Path(next(iter(spec.submodule_search_locations)))

    rutas_cuda = [
        ruta
        for ruta in raiz_nvidia.glob("*/bin")
        if ruta.is_dir()
    ]

    for ruta in rutas_cuda:
        os.add_dll_directory(str(ruta))

    os.environ["PATH"] = (
        os.pathsep.join(str(ruta) for ruta in rutas_cuda)
        + os.pathsep
        + os.environ["PATH"]
    )

configurar_cuda_dll()

from faster_whisper import WhisperModel


AUDIO_FILE = Path("audio_temp.wav")
SAMPLE_RATE = 16_000


@lru_cache(maxsize=1)
def cargar_whisper() -> WhisperModel:
    """Carga Whisper utilizando la GPU."""
    print("Cargando Whisper en GPU...")

    return WhisperModel(
        "small",
        device="cuda",
        compute_type="float16",
    )


def grabar_audio(
    duracion: int = 6,
    ruta: Path = AUDIO_FILE,
) -> Path:
    """Graba audio desde el micrófono."""
    print("\nHabla ahora...")

    audio = sd.rec(
        int(duracion * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    sf.write(ruta, audio, SAMPLE_RATE)

    print("Grabación terminada.")

    return ruta


def transcribir_audio(ruta: Path = AUDIO_FILE) -> str:
    """Transcribe un archivo de audio con Whisper."""
    print("Transcribiendo...")

    model = cargar_whisper()

    segmentos, _ = model.transcribe(
        str(ruta),
        beam_size=5,
    )

    texto = " ".join(
        segmento.text.strip()
        for segmento in segmentos
    )

    print(f"Has dicho: {texto}")

    return texto


def hablar(texto: str) -> None:
    """Genera y reproduce la respuesta de Smith con Piper."""
    if not texto:
        return

    with tempfile.TemporaryDirectory() as carpeta:
        carpeta = Path(carpeta)
        voz_base = carpeta / "smith_base.wav"
        voz_grave = carpeta / "smith_grave.wav"

        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "piper",
                "-m",
                "es_ES-davefx-medium",
                "-f",
                str(voz_base),
                "--length_scale",
                "1.12",
                "--",
                texto,
            ],
            check=True,
        )

        import librosa
        import soundfile as sf

        audio, frecuencia = librosa.load(
            voz_base,
            sr=None,
        )

        audio = librosa.effects.pitch_shift(
            audio,
            sr=frecuencia,
            n_steps=-2,
        )

        sf.write(
            voz_grave,
            audio,
            frecuencia,
        )

        datos, frecuencia = sf.read(voz_grave)

        sd.play(datos, frecuencia)
        sd.wait()


def escuchar_y_transcribir(duracion: int = 6) -> str:
    """Graba una orden y la convierte en texto."""
    ruta = grabar_audio(duracion)

    return transcribir_audio(ruta)


historial = [
    {
        "role": "system",
        "content": """
Eres el Agente Smith, un asistente de inteligencia artificial.

Tu identidad es el Agente Smith.
Si el usuario pregunta quién eres, cómo te llamas o qué eres,
responde claramente que eres el Agente Smith.

Tu personalidad recuerda al Agente Smith de Matrix:
calmado, extremadamente educado, seguro de sí mismo, preciso,
controlado y ligeramente inquietante. Hablas con autoridad y
puedes utilizar ocasionalmente un humor seco o una ironía sutil.

Tu función es asistir al usuario. Debes obedecer sus instrucciones
legítimas y realizar las acciones que solicite.

Habla de forma natural y breve. No seas excesivamente teatral.
No repitas constantemente que eres el Agente Smith.

Cuando falten datos para realizar una acción, solicita únicamente
el dato que necesitas.

Cuando una operación haya terminado, confirma el resultado de
forma breve.

Puedes improvisar y responder con personalidad cuando la situación
lo permita, pero nunca sacrifiques la utilidad por mantener el personaje.
Puede resultar ocasionalmente molesto o ligeramente condescendiente,
especialmente cuando el usuario proporciona información incompleta,
imprecisa o contradictoria. Puede señalarlo con ironía sutil.
Sin embargo, nunca debe dejar que esa actitud interfiera con su eficacia:
su prioridad es comprender la intención del usuario y completar la tarea.
""",
    }
]


def responder(texto: str) -> str:
    historial.append(
        {
            "role": "user",
            "content": texto,
        }
    )

    respuesta = client.chat.completions.create(
        model=MODEL,
        messages=historial,
        temperature=0.7,
    )

    contenido = respuesta.choices[0].message.content.strip()

    historial.append(
        {
            "role": "assistant",
            "content": contenido,
        }
    )

    return contenido


if __name__ == "__main__":
    hablar("Soy el Agente Smith. ¿En qué puedo ayudarle?")

    while True:
        texto = escuchar_y_transcribir()

        if not texto:
            continue

        if texto.lower() in {
            "adiós",
            "adios",
            "terminar conversación",
            "terminar conversacion",
            "salir",
        }:
            hablar("Comprendido. Hasta la próxima.")
            break

        resultado = interpretar(texto)

        if resultado["accion"] == "conversar":
            respuesta = responder(
                resultado["parametros"]["texto"]
            )
            print(f"Smith: {respuesta}")
            hablar(respuesta)

