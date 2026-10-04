from functools import lru_cache
from pathlib import Path

import pyttsx3
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


AUDIO_FILE = Path("audio_temp.wav")
SAMPLE_RATE = 16_000
SILENCE_DURATION = 1.5


@lru_cache(maxsize=1)
def cargar_whisper() -> WhisperModel:
    """Carga Whisper usando GPU y utiliza CPU como alternativa."""
    try:
        print(" Cargando Whisper en GPU...")
        return WhisperModel(
            "small",
            device="cuda",
            compute_type="float16",
        )
    except Exception as error:
        print(f"No se ha podido usar CUDA: {error}")
        print(" Cargando Whisper en CPU...")
        return WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
        )


def grabar_audio(
    duracion: int = 6,
    ruta: Path = AUDIO_FILE,
) -> Path:
    """Graba audio desde el micrófono."""
    print("\n Habla ahora...")

    audio = sd.rec(
        int(duracion * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    sf.write(ruta, audio, SAMPLE_RATE)

    print(" Grabación terminada.")

    return ruta


def transcribir_audio(ruta: Path = AUDIO_FILE) -> str:
    """Transcribe un archivo de audio con Whisper."""
    print(" Transcribiendo...")

    model = cargar_whisper()

    segmentos, _ = model.transcribe(
        str(ruta),
        beam_size=5,
    )

    texto = " ".join(
        segmento.text.strip()
        for segmento in segmentos
    )

    print(f" Has dicho: {texto}")

    return texto


def hablar(texto: str) -> None:
    """Reproduce una respuesta mediante síntesis de voz."""
    if not texto:
        return

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    engine.say(texto)
    engine.runAndWait()


def escuchar_y_transcribir(duracion: int = 6) -> str:
    """Graba una orden y la convierte directamente a texto."""
    ruta = grabar_audio(duracion)

    return transcribir_audio(ruta)


if __name__ == "__main__":
    texto = escuchar_y_transcribir()

    if texto:
        hablar(f"Has dicho: {texto}")