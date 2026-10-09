from src.agente.asistente import ejecutar as ejecutar_voz
from src.agente.interprete import reiniciar_historial, responder
from src.agente.memoria import cargar_memoria


def ejecutar_escrito() -> None:
    """Ejecuta una conversación escrita con el Agente Smith."""
    memoria = cargar_memoria()
    reiniciar_historial()

    print("\nAgente Smith activo.")
    print("Escriba «adios» para volver al menú.\n")

    while True:
        texto = input("Tú: ").strip()

        if not texto:
            continue

        if texto.casefold() in {"adios", "adiós"}:
            print("Smith: Hasta pronto.")
            return

        respuesta = responder(
            texto,
            memoria,
        )

        print(f"Smith: {respuesta}")


def menu() -> None:
    """Muestra el menú principal del Agente Smith."""
    while True:
        print("\n" + "=" * 42)
        print("              AGENTE SMITH")
        print("=" * 42)
        print("1. Interacción escrita")
        print("2. Interacción por voz")
        print("3. Salir")
        print("=" * 42)

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            ejecutar_escrito()

        elif opcion == "2":
            ejecutar_voz()

        elif opcion == "3":
            print("\nAgente Smith finalizado.")
            return

        else:
            print("\nOpción no válida.")


if __name__ == "__main__":
    menu()
