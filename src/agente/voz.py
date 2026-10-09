import gc
import importlib.util
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from qwen_tts import Qwen3TTSModel

from .config import (
    FRECUENCIA_MUESTREO,
    MODELO_TTS,
    MODELO_WHISPER,
    PALABRAS_ACTIVACION,
    PALABRAS_PARADA,
    RUTA_AUDIO_TEMPORAL,
    RUTA_REFERENCIA_VOZ,
    UMBRAL_RMS,
)


def configurar_cuda_dll() -> None:
    """Configura las DLL de CUDA instaladas mediante pip."""
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

    if rutas_cuda:
        os.environ["PATH"] = (
            os.pathsep.join(str(ruta) for ruta in rutas_cuda)
            + os.pathsep
            + os.environ["PATH"]
        )


configurar_cuda_dll()


@lru_cache(maxsize=1)
def cargar_whisper() -> WhisperModel:
    """Carga Whisper en la GPU."""
    print("Cargando Whisper en GPU...")

    return WhisperModel(
        MODELO_WHISPER,
        device="cuda",
        compute_type="float16",
    )


def reconocer_audio(ruta: str | Path) -> str:
    """Transcribe un archivo de audio con el Whisper compartido."""
    model = cargar_whisper()

    try:
        segmentos, _ = model.transcribe(
            str(ruta),
            language="es",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        textos = [
            segmento.text.strip()
            for segmento in segmentos
            if segmento.no_speech_prob < 0.6
            and segmento.text.strip()
        ]

        return " ".join(textos)

    except (OSError, RuntimeError) as error:
        print(f"Error al transcribir el audio: {error}")
        return ""

@lru_cache(maxsize=1)
def cargar_tts() -> Qwen3TTSModel:
    """Carga Qwen3-TTS en la GPU."""
    print("Cargando Qwen3-TTS en GPU...")

    return Qwen3TTSModel.from_pretrained(
        MODELO_TTS,
        device_map="cuda",
        dtype=torch.bfloat16,
    )


def liberar_whisper() -> None:
    """Libera Whisper de la GPU."""
    cargar_whisper.cache_clear()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def liberar_tts() -> None:
    """Libera Qwen3-TTS de la GPU."""
    cargar_tts.cache_clear()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def grabar_audio(
    duracion_maxima: int = 10,
    ruta: Path = RUTA_AUDIO_TEMPORAL,
    mostrar_prompt: bool = True,
    espera_inicial: float = 3.0,
) -> Path | None:
    """Graba audio hasta detectar silencio prolongado."""
    bloque = 320
    duracion_bloque = bloque / FRECUENCIA_MUESTREO

    silencio_necesario = int(2.0 / duracion_bloque)
    espera_maxima = int(espera_inicial / duracion_bloque)
    bloques_maximos = int(duracion_maxima / duracion_bloque)
    bloques_preroll = int(0.3 / duracion_bloque)

    buffer_preroll: list[np.ndarray] = []
    bloques_audio: list[np.ndarray] = []

    hablando = False
    bloques_silencio = 0
    bloques_espera = 0

    if mostrar_prompt:
        print("\nHabla ahora...")

    try:
        with sd.InputStream(
            samplerate=FRECUENCIA_MUESTREO,
            channels=1,
            dtype="float32",
            blocksize=bloque,
        ) as stream:
            for _ in range(bloques_maximos):
                audio, _ = stream.read(bloque)

                audio = np.asarray(
                    audio[:, 0],
                    dtype=np.float32,
                )

                rms = float(
                    np.sqrt(
                        np.mean(
                            np.square(audio)
                        )
                    )
                )

                if not hablando:
                    buffer_preroll.append(audio)

                    if len(buffer_preroll) > bloques_preroll:
                        buffer_preroll.pop(0)

                    bloques_espera += 1

                    if rms > UMBRAL_RMS:
                        hablando = True
                        bloques_audio.extend(buffer_preroll)
                        bloques_silencio = 0

                    elif bloques_espera >= espera_maxima:
                        if mostrar_prompt:
                            print("No se detectó voz.")
                        return None

                    continue

                bloques_audio.append(audio)

                if rms <= UMBRAL_RMS:
                    bloques_silencio += 1
                else:
                    bloques_silencio = 0

                if bloques_silencio >= silencio_necesario:
                    break

    except sd.PortAudioError as error:
        print(f"No se pudo acceder al micrófono: {error}")
        return None

    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error durante la grabación: {error}")
        return None

    if not bloques_audio:
        if mostrar_prompt:
            print("No se detectó voz.")
        return None

    audio_final = np.concatenate(bloques_audio)

    if mostrar_prompt:
        print("Grabación terminada.")

    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            ruta,
            audio_final,
            FRECUENCIA_MUESTREO,
        )

    except (OSError, RuntimeError) as error:
        print(f"No se pudo guardar el audio: {error}")
        return None

    return ruta


def transcribir_audio(
    ruta: Path = RUTA_AUDIO_TEMPORAL,
    mostrar_resultado: bool = True,
) -> str:
    """Transcribe audio utilizando el reconocedor compartido."""
    if mostrar_resultado:
        print("Transcribiendo...")

    texto_final = reconocer_audio(ruta)

    if not texto_final:
        if mostrar_resultado:
            print("No se ha detectado una frase válida.")
        return ""

    if mostrar_resultado:
        print(f"Has dicho: {texto_final}")

    return texto_final




def escuchar_y_transcribir(timeout: float = 3.0) -> str:
    """Graba una frase y la transcribe.

    timeout: segundos máximos esperando a que el usuario empiece a hablar.
    """
    ruta = grabar_audio(
        duracion_maxima=int(timeout) + 10,
        espera_inicial=timeout,
    )

    if ruta is None:
        return ""

    return transcribir_audio(ruta)

def _normalizar_texto(texto: str) -> str:
    """Normaliza mayúsculas, acentos y espacios para comparar órdenes."""
    texto = unicodedata.normalize("NFKD", texto.casefold())
    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )
    return re.sub(r"[^a-z0-9]+", " ", texto).strip()


def _contiene_palabra(texto: str, palabras: tuple[str, ...]) -> bool:
    """Comprueba órdenes completas, sin coincidencias dentro de otras palabras."""
    texto_normalizado = f" {_normalizar_texto(texto)} "

    return any(
        f" {_normalizar_texto(palabra)} " in texto_normalizado
        for palabra in palabras
        if _normalizar_texto(palabra)
    )


def esperar_activacion() -> bool:
    """Espera hasta detectar la palabra de activación."""
    while True:
        ruta = grabar_audio(
            duracion_maxima=5,
            mostrar_prompt=False,
        )

        if ruta is None:
            continue

        texto = transcribir_audio(
            ruta,
            mostrar_resultado=False,
        )

        if _contiene_palabra(texto, PALABRAS_ACTIVACION):
            print("Activación detectada.")
            return True


def es_parada(texto: str) -> bool:
    """Comprueba si el usuario ha solicitado terminar la conversación."""
    return _contiene_palabra(texto, PALABRAS_PARADA)


def hablar(
    texto: str,
    idioma: str = "Spanish",
) -> None:
    """Genera y reproduce una respuesta con la voz de Smith."""
    if not texto:
        return

    if not RUTA_REFERENCIA_VOZ.exists():
        raise FileNotFoundError(
            f"No existe la referencia de voz: {RUTA_REFERENCIA_VOZ}"
        )

    liberar_whisper()

    model = cargar_tts()

    try:
        wavs, frecuencia = model.generate_voice_clone(
            text=texto,
            language=idioma,
            ref_audio=str(RUTA_REFERENCIA_VOZ),
            x_vector_only_mode=True,
            max_new_tokens=4096,
        )

        datos = wavs[0]

        sd.play(
            datos,
            frecuencia,
        )

        sd.wait()

    except (RuntimeError, OSError) as error:
        print(
            "Error al generar o reproducir la voz: "
            f"{error}"
        )

    finally:
        del model
        liberar_tts()

