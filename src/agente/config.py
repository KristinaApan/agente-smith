from pathlib import Path

# Directorio raíz del proyecto.
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

# Datos persistentes.
DIRECTORIO_DATOS = RAIZ_PROYECTO / "data"
RUTA_MEMORIA = DIRECTORIO_DATOS / "memoria" / "memoria.json"
RUTA_AUDIO_TEMPORAL = DIRECTORIO_DATOS / "voz" / "audio_temp.wav"
RUTA_REFERENCIA_VOZ = DIRECTORIO_DATOS / "voz" / "smith_reference.wav"

# LM Studio.
LMSTUDIO_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"

# Modelos locales.
MODELO_LLM = "qwen3-8b"
MODELO_VISION = "qwen3-vl-8b-instruct"
MODELO_WHISPER = "small"
MODELO_TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Audio.
FRECUENCIA_MUESTREO = 16_000
UMBRAL_RMS = 0.005

# Activación y parada.
PALABRAS_ACTIVACION = ("smith", "esmit", "smit", "esmís")
PALABRAS_PARADA = (
    "stop",
    "para",
    "detente",
    "cállate",
    "adiós",
    "adios",
)
