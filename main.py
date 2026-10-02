import base64

import json

import requests

from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


def traducir(texto_json: str) -> str:
    datos = json.loads(texto_json)

    texto = datos["texto"]
    idioma = datos["idioma"]

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {
                "role": "user",
                "content": f"Traduce este texto al {idioma}:\n\n{texto}",
            }
        ],
    )

    return response.choices[0].message.content


def descargar_imagen(url: str, ruta: str) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(ruta, "wb") as archivo:
        archivo.write(response.content)

    print(f"Imagen guardada en: {ruta}")


def resolver_problema_png(ruta: str) -> str:
    with open(ruta, "rb") as archivo:
        imagen = base64.b64encode(archivo.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="qwen3-vl-8b-instruct",
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
                            "url": f"data:image/png;base64,{imagen}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


print(resolver_problema_png("problema.png"))

def menu() -> None:
    while True:
        print()
        print("         ASISTENTE IA")
        print()
        print("1. Traducir texto")
        print("2. Descargar imagen")
        print("3. Resolver problema desde PNG")
        print("4. Transcribir audio")
        print("5. Salir")

        opcion = input("\nSelecciona una opción: ")

        if opcion == "1":
            entrada = input(
                'Introduce el JSON (ejemplo: {"texto": "Hola", "idioma": "inglés"}):\n'
            )

            resultado = traducir(entrada)

            print("\nTraducción:")
            print(resultado)

        elif opcion == "2":
            url = input("\nIntroduce la URL de la imagen: ")
            ruta = input("Introduce la ruta donde quieres guardarla: ")

            descargar_imagen(url, ruta)

        elif opcion == "3":
            ruta = input("\nIntroduce la ruta del archivo PNG: ")
            resultado = resolver_problema_png(ruta)

            print("\nResultado:")
            print(resultado)

        elif opcion == "4":
            print("\nEsta opción estará disponible próximamente.")

        elif opcion == "5":
            print("\nSaliendo del asistente...")
            break

        else:
            print("\nOpción no válida.")




if __name__ == "__main__":
    menu()