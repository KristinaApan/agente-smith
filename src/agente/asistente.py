from .interprete import generar_saludo, reiniciar_historial, responder
from .memoria import cargar_memoria
from .voz import (
    es_parada,
    escuchar_y_transcribir,
    esperar_activacion,
    hablar,
)

VENTANA_CONVERSACION = 8  # segundos de escucha sin palabra de activación


def _escuchar(timeout: float | None) -> str:
    """Escucha esperando como máximo `timeout` segundos a que hable."""
    if timeout is None:
        return escuchar_y_transcribir()

    return escuchar_y_transcribir(timeout=timeout)
    

def ejecutar() -> None:
    """Ejecuta el ciclo principal del Agente Smith."""
    memoria = cargar_memoria()
    reiniciar_historial()

    while True:
        print('\nEsperando: «Agente Smith»...')

        if not esperar_activacion():
            continue

        hablar(generar_saludo())

        # Primera petición: sin límite estricto.
        texto = _escuchar(None)

        # Conversación: mientras haya texto dentro de la ventana.
        while texto:
            if es_parada(texto):
                hablar("Hasta pronto.")
                return

            print(f"Tú: {texto}")

            respuesta = responder(
                texto,
                memoria,
                hablar=hablar,
            )

            print(f"Smith: {respuesta}")
            hablar(respuesta)

            texto = _escuchar(VENTANA_CONVERSACION)

        print("\nSin actividad. Vuelvo a la espera.")