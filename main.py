import json

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


entrada = '{"texto": "Hola, ¿cómo estás?", "idioma": "inglés"}'

resultado = traducir(entrada)

print(resultado)