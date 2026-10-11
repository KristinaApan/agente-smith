import json
from pathlib import Path
from typing import Any

RUTA_MEMORIA = Path("data/memoria/memoria.json")


def cargar_memoria() -> dict[str, Any]:
    """Carga la memoria persistente de Smith desde un archivo JSON."""
    if not RUTA_MEMORIA.exists():
        return {"recuerdos": []}

    try:
        datos = json.loads(RUTA_MEMORIA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"recuerdos": []}

    if not isinstance(datos, dict):
        return {"recuerdos": []}

    recuerdos = datos.get("recuerdos")

    if not isinstance(recuerdos, list):
        datos["recuerdos"] = []

    return datos


def guardar_memoria(memoria: dict[str, Any]) -> None:
    """Guarda la memoria persistente de Smith en formato JSON."""
    RUTA_MEMORIA.parent.mkdir(parents=True, exist_ok=True)

    temporal = RUTA_MEMORIA.with_suffix(".tmp")

    try:
        temporal.write_text(
            json.dumps(
                memoria,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporal.replace(RUTA_MEMORIA)
    except OSError:
        if temporal.exists():
            temporal.unlink()
        raise


def anadir_recuerdo(recuerdo: str) -> None:
    """Añade un recuerdo a la memoria persistente."""
    recuerdo = recuerdo.strip()

    if not recuerdo:
        raise ValueError("El recuerdo no puede estar vacío.")

    memoria = cargar_memoria()
    recuerdos = memoria.setdefault("recuerdos", [])

    if recuerdo not in recuerdos:
        recuerdos.append(recuerdo)
        guardar_memoria(memoria)


def eliminar_recuerdo(recuerdo: str) -> bool:
    """Elimina un recuerdo y devuelve True si existía."""
    recuerdo = recuerdo.strip()

    if not recuerdo:
        return False

    memoria = cargar_memoria()
    recuerdos = memoria["recuerdos"]

    if recuerdo not in recuerdos:
        return False

    recuerdos.remove(recuerdo)
    guardar_memoria(memoria)
    return True


def obtener_recuerdos() -> list[str]:
    """Devuelve los recuerdos persistentes de Smith."""
    memoria = cargar_memoria()
    return list(memoria["recuerdos"])


def limpiar_memoria() -> None:
    """Elimina todos los recuerdos persistentes."""
    guardar_memoria({"recuerdos": []})
