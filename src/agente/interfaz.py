
import queue
import random
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import FRECUENCIA_MUESTREO, RUTA_AUDIO_TEMPORAL
from .interprete import reiniciar_historial, responder
from .memoria import cargar_memoria
from .voz import hablar, reconocer_audio

ANCHO = 900
ALTO = 560
VERDE = "#00ff66"
FONDO = "#020604"
VERDE_OSCURO = "#00883b"
ROJO = "#ff4141"
TITULO = "AGENTE SMITH"
BLOQUE_AUDIO = 320
DURACION_MAXIMA = 120


class InterfazSmith:
    """Interfaz gráfica de Smith con grabación manual."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Agente Smith")
        self.root.geometry(f"{ANCHO}x{ALTO}")
        self.root.minsize(650, 440)
        self.root.configure(bg=FONDO)

        self.canvas = tk.Canvas(
            root,
            bg=FONDO,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.columnas: list[dict[str, int]] = []
        self.indice = 0
        self.fase = "lluvia"
        self.cola: queue.Queue[tuple[str, object]] = queue.Queue()
        self.detener_grabacion = threading.Event()
        self.estado = "listo"
        self.cerrando = False
        self.ventana_panel: int | None = None
        self.id_titulo: int | None = None

        self.fila_solicitud: tk.Frame | None = None
        self.entrada_solicitud: tk.Entry | None = None

        self.memoria = cargar_memoria()
        reiniciar_historial()

        self.root.bind("<Escape>", lambda _event: self.cerrar())
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.canvas.bind("<Configure>", self.redimensionar)

        self.iniciar_lluvia()
        self.root.after(100, self.procesar_cola)

    def iniciar_lluvia(self) -> None:
        """Muestra la lluvia de caracteres verdes al iniciar."""
        self.fase = "lluvia"
        self.canvas.delete("all")
        self.columnas.clear()

        ancho = max(self.canvas.winfo_width(), ANCHO)
        alto = max(self.canvas.winfo_height(), ALTO)

        for x in range(12, ancho, 22):
            self.columnas.append(
                {
                    "x": x,
                    "y": random.randint(-alto, 0),
                    "velocidad": random.randint(3, 11),
                    "longitud": random.randint(5, 15),
                }
            )

        self.animar_lluvia(0)

    def animar_lluvia(self, cuadro: int) -> None:
        if self.fase != "lluvia":
            return

        self.canvas.delete("lluvia")
        alto = self.canvas.winfo_height()

        for columna in self.columnas:
            x, y = columna["x"], columna["y"]

            for paso in range(columna["longitud"]):
                color = VERDE if paso == 0 else VERDE_OSCURO
                self.canvas.create_text(
                    x,
                    y - paso * 18,
                    text=random.choice("0123456789ABCDEF"),
                    fill=color,
                    font=("Consolas", 13, "bold"),
                    tags="lluvia",
                )

            columna["y"] += columna["velocidad"]

            if columna["y"] - columna["longitud"] * 18 > alto:
                columna["y"] = random.randint(-220, -20)

        if cuadro < 65:
            self.root.after(
                30,
                lambda: self.animar_lluvia(cuadro + 1),
            )
        else:
            self.fase = "escribiendo"
            self.canvas.delete("all")
            self.indice = 0
            self.escribir_titulo()

    def escribir_titulo(self) -> None:
        if self.fase != "escribiendo":
            return

        self.canvas.delete("titulo", "cursor")
        ancho = self.canvas.winfo_width()
        alto = self.canvas.winfo_height()

        self.id_titulo = self.canvas.create_text(
            ancho // 2,
            alto // 2,
            text=TITULO[:self.indice],
            fill=VERDE,
            font=("Consolas", 34, "bold"),
            tags="titulo",
        )

        if self.indice < len(TITULO):
            self.indice += 1
            self.root.after(115, self.escribir_titulo)
        else:
            self.fase = "cursor"
            self.parpadear_cursor(True)
            self.root.after(700, self.mostrar_menu)

    def parpadear_cursor(self, visible: bool) -> None:
        if self.fase != "cursor":
            return

        self.canvas.delete("cursor")

        if visible:
            ancho = self.canvas.winfo_width()
            alto = self.canvas.winfo_height()

            self.canvas.create_text(
                ancho // 2 + 235,
                alto // 2,
                text="▌",
                fill=VERDE,
                font=("Consolas", 34, "bold"),
                tags="cursor",
            )

        self.root.after(
            300,
            lambda: self.parpadear_cursor(not visible),
        )

    def mostrar_menu(self) -> None:
        """Muestra el panel de conversación tras la animación."""
        if self.fase != "cursor":
            return

        self.fase = "menu"
        self.canvas.delete("all")

        self.id_titulo = self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            48,
            text=TITULO,
            fill=VERDE,
            font=("Consolas", 27, "bold"),
            tags="titulo",
        )

        self.panel = tk.Frame(
            self.canvas,
            bg=FONDO,
            padx=12,
            pady=8,
        )

        self.estado_label = tk.Label(
            self.panel,
            text="Sistema preparado",
            bg=FONDO,
            fg=VERDE,
            font=("Consolas", 11),
        )
        self.estado_label.pack(pady=(0, 8))

        self.conversacion = ScrolledText(
            self.panel,
            height=8,
            wrap="word",
            bg="#07110b",
            fg="#b8ffc9",
            insertbackground=VERDE,
            font=("Consolas", 10),
            relief="flat",
            state="disabled",
        )
        self.conversacion.pack(
            fill="both",
            expand=True,
            pady=(0, 8),
        )

        self.fila_texto = tk.Frame(self.panel, bg=FONDO)
        self.fila_texto.pack(fill="x", pady=(0, 8))

        self.entrada = tk.Entry(
            self.fila_texto,
            bg="#07110b",
            fg=VERDE,
            insertbackground=VERDE,
            font=("Consolas", 11),
            relief="flat",
        )
        self.entrada.pack(
            side="left",
            fill="x",
            expand=True,
            ipady=7,
        )
        self.entrada.bind("<Return>", self.enviar_texto)

        self.boton_enviar = tk.Button(
            self.fila_texto,
            text="Enviar",
            command=self.enviar_texto,
            bg="#063b1a",
            fg=VERDE,
            activebackground="#0b5528",
            activeforeground="white",
            relief="flat",
            padx=14,
        )
        self.boton_enviar.pack(side="left", padx=(8, 0))

        fila_botones = tk.Frame(self.panel, bg=FONDO)
        fila_botones.pack(fill="x")

        self.boton_voz = tk.Button(
            fila_botones,
            text="●  EMPEZAR GRABACIÓN",
            command=self.alternar_grabacion,
            bg="#063b1a",
            fg=VERDE,
            activebackground="#0b5528",
            activeforeground="white",
            font=("Consolas", 11, "bold"),
            relief="flat",
            pady=10,
        )
        self.boton_voz.pack(
            side="left",
            fill="x",
            expand=True,
        )

        self.boton_salir = tk.Button(
            fila_botones,
            text="Salir",
            command=self.cerrar,
            bg="#171717",
            fg="#dddddd",
            activebackground="#333333",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
        )
        self.boton_salir.pack(side="left", padx=(8, 0))

        self.ventana_panel = self.canvas.create_window(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2 + 45,
            window=self.panel,
            width=max(520, self.canvas.winfo_width() - 80),
            anchor="center",
            tags="panel",
        )

        self.entrada.focus_set()
        self.redimensionar()

    def redimensionar(
        self,
        _evento: tk.Event | None = None,
    ) -> None:
        if self.fase != "menu":
            return

        ancho = self.canvas.winfo_width()
        alto = self.canvas.winfo_height()

        if self.id_titulo is not None:
            self.canvas.coords(self.id_titulo, ancho // 2, 48)

        if self.ventana_panel is not None:
            self.canvas.coords(
                self.ventana_panel,
                ancho // 2,
                alto // 2 + 45,
            )
            self.canvas.itemconfigure(
                self.ventana_panel,
                width=max(520, ancho - 80),
            )

    def agregar_mensaje(self, autor: str, texto: str) -> None:
        self.conversacion.configure(state="normal")
        self.conversacion.insert("end", f"{autor}: {texto}\n\n")
        self.conversacion.see("end")
        self.conversacion.configure(state="disabled")

    def cambiar_estado(self, estado: str) -> None:
        self.estado_label.configure(text=estado)

        if estado == "grabando":
            self.estado = "grabando"
            self.boton_voz.configure(
                text="■  DETENER GRABACIÓN",
                bg="#591515",
                fg=ROJO,
                state="normal",
            )
        elif estado == "procesando":
            self.estado = "procesando"
            self.boton_voz.configure(
                text="PROCESANDO...",
                bg="#17291c",
                fg=VERDE,
                state="disabled",
            )
        elif estado == "listo":
            self.estado = "listo"
            self.boton_voz.configure(
                text="●  EMPEZAR GRABACIÓN",
                bg="#063b1a",
                fg=VERDE,
                state="normal",
            )
            self.boton_enviar.configure(state="normal")
            self.entrada.configure(state="normal")

    def alternar_grabacion(self) -> None:
        if self.estado == "grabando":
            self.detener_grabacion.set()
            self.boton_voz.configure(
                text="CERRANDO GRABACIÓN...",
                state="disabled",
            )
            self.estado_label.configure(text="Guardando audio...")
            return

        if self.estado != "listo":
            return

        self.detener_grabacion.clear()
        self.boton_enviar.configure(state="disabled")
        self.entrada.configure(state="disabled")
        self.cambiar_estado("grabando")
        self.estado_label.configure(
            text="Grabando... pulsa otra vez para detener."
        )

        threading.Thread(
            target=self.grabar_y_procesar,
            daemon=True,
            name="smith-audio",
        ).start()

    def grabar_y_procesar(self) -> None:
        """Graba hasta el segundo clic y procesa el audio en segundo plano."""
        try:
            bloques: list[np.ndarray] = []
            max_bloques = int(
                DURACION_MAXIMA
                * FRECUENCIA_MUESTREO
                / BLOQUE_AUDIO
            )

            with sd.InputStream(
                samplerate=FRECUENCIA_MUESTREO,
                channels=1,
                dtype="float32",
                blocksize=BLOQUE_AUDIO,
            ) as stream:
                for _ in range(max_bloques):
                    if self.detener_grabacion.is_set():
                        break

                    audio, _ = stream.read(BLOQUE_AUDIO)
                    bloques.append(
                        np.asarray(audio[:, 0], dtype=np.float32).copy()
                    )

            if not bloques:
                self.cola.put(("error", "No se ha grabado audio."))
                return

            ruta = RUTA_AUDIO_TEMPORAL
            ruta.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                ruta,
                np.concatenate(bloques),
                FRECUENCIA_MUESTREO,
            )

            self.cola.put(("estado", "procesando"))
            self.cola.put(
                ("estado_texto", "Transcribiendo con Whisper...")
            )

            texto = reconocer_audio(ruta)

            if not texto:
                self.cola.put(
                    ("aviso", "No se ha detectado una frase válida.")
                )
                return

            self.cola.put(("mensaje", ("Tú", texto)))
            self.cola.put(
                ("estado_texto", "Smith está preparando la respuesta...")
            )

            respuesta = responder(
                texto,
                self.memoria,
                hablar=hablar,
                solicitar_entrada=self.solicitar_entrada,
            )

            if self.cerrando:
                return

            self.cola.put(("mensaje", ("Smith", respuesta)))
            self.cola.put(
                ("estado_texto", "Smith está respondiendo...")
            )
            hablar(respuesta)

        except (
            sd.PortAudioError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            self.cola.put(
                ("error", f"Error de grabación o procesamiento: {error}")
            )
        # La interfaz debe comunicar también los errores inesperados.
        except Exception as error:  # noqa: BLE001
            self.cola.put(("error", f"Error inesperado: {error}"))
        finally:
            self.cola.put(("estado", "listo"))

    def enviar_texto(
        self,
        _evento: tk.Event | None = None,
    ) -> None:
        if self.estado != "listo":
            return

        texto = self.entrada.get().strip()
        if not texto:
            return

        if texto.casefold() in {"adios", "adiós"}:
            self.agregar_mensaje("Smith", "Hasta pronto.")
            self.entrada.delete(0, "end")
            return

        self.entrada.delete(0, "end")
        self.agregar_mensaje("Tú", texto)
        self.boton_enviar.configure(state="disabled")
        self.entrada.configure(state="disabled")
        self.boton_voz.configure(state="disabled")
        self.estado = "procesando"
        self.estado_label.configure(text="Smith está pensando...")

        threading.Thread(
            target=self.responder_texto,
            args=(texto,),
            daemon=True,
            name="smith-texto",
        ).start()

    def responder_texto(self, texto: str) -> None:
        """Procesa un mensaje escrito sin generar voz."""
        try:
            respuesta = responder(
                texto,
                self.memoria,
                hablar=None,
                solicitar_entrada=self.solicitar_entrada,
            )

            if self.cerrando:
                return

            self.cola.put(("mensaje", ("Smith", respuesta)))

        # La interfaz debe comunicar también los errores inesperados.
        except Exception as error:  # noqa: BLE001
            if not self.cerrando:
                self.cola.put(
                    (
                        "error",
                        f"No se pudo generar la respuesta: {error}",
                    )
                )
        finally:
            if not self.cerrando:
                self.cola.put(("estado", "listo"))

    def solicitar_entrada(self, pregunta: str) -> str | None:
        """Solicita un dato dentro de la interfaz desde un hilo de trabajo."""
        if self.cerrando:
            return None

        evento = threading.Event()
        resultado: dict[str, str | None] = {"valor": None}

        self.cola.put(
            ("solicitar_entrada", (pregunta, evento, resultado))
        )

        while not evento.wait(0.1):
            if self.cerrando:
                return None

        return resultado["valor"]

    def mostrar_solicitud_entrada(
        self,
        pregunta: str,
        evento: threading.Event,
        resultado: dict[str, str | None],
    ) -> None:
        """Muestra el campo de solicitud integrado en el panel principal."""
        if self.cerrando:
            resultado["valor"] = None
            evento.set()
            return

        if self.fila_solicitud is not None:
            self.agregar_mensaje(
                "Error",
                "Ya existe una solicitud de entrada pendiente.",
            )
            resultado["valor"] = None
            evento.set()
            return

        fila = tk.Frame(self.panel, bg=FONDO)
        self.fila_solicitud = fila

        etiqueta = tk.Label(
            fila,
            text=pregunta,
            bg=FONDO,
            fg=VERDE,
            font=("Consolas", 10),
            anchor="w",
            justify="left",
            wraplength=700,
        )
        etiqueta.pack(fill="x", pady=(0, 5))

        entrada = tk.Entry(
            fila,
            bg="#07110b",
            fg=VERDE,
            insertbackground=VERDE,
            font=("Consolas", 11),
            relief="flat",
        )
        self.entrada_solicitud = entrada
        entrada.pack(fill="x", ipady=7, pady=(0, 6))

        fila_botones = tk.Frame(fila, bg=FONDO)
        fila_botones.pack(fill="x")

        def finalizar(valor: str | None) -> None:
            if self.cerrando:
                valor = None

            resultado["valor"] = valor

            if self.fila_solicitud is not None:
                self.fila_solicitud.destroy()

            self.fila_solicitud = None
            self.entrada_solicitud = None
            evento.set()

            if not self.cerrando:
                self.entrada.focus_set()

        boton_confirmar = tk.Button(
            fila_botones,
            text="Confirmar",
            command=lambda: finalizar(entrada.get().strip()),
            bg="#063b1a",
            fg=VERDE,
            activebackground="#0b5528",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=5,
        )
        boton_confirmar.pack(side="left")

        boton_cancelar = tk.Button(
            fila_botones,
            text="Cancelar",
            command=lambda: finalizar(None),
            bg="#171717",
            fg="#dddddd",
            activebackground="#333333",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=5,
        )
        boton_cancelar.pack(side="left", padx=(8, 0))

        entrada.bind(
            "<Return>",
            lambda _event: finalizar(entrada.get().strip()),
        )

        fila.pack(
            fill="x",
            pady=(0, 8),
            before=self.fila_texto,
        )
        entrada.focus_set()

    def procesar_cola(self) -> None:
        """Procesa los mensajes pendientes de los hilos de trabajo."""
        if self.cerrando:
            return

        try:
            while True:
                tipo, contenido = self.cola.get_nowait()

                if tipo == "solicitar_entrada":
                    pregunta, evento, resultado = contenido
                    self.mostrar_solicitud_entrada(
                        pregunta,
                        evento,
                        resultado,
                    )

                elif tipo == "mensaje":
                    autor, texto = contenido
                    self.agregar_mensaje(autor, texto)

                elif tipo == "estado":
                    if contenido in {"grabando", "procesando", "listo"}:
                        self.cambiar_estado(contenido)

                elif tipo == "estado_texto":
                    self.estado_label.configure(text=str(contenido))

                elif tipo == "error":
                    self.agregar_mensaje("Error", str(contenido))
                    self.estado_label.configure(
                        text="Se ha producido un error."
                    )

                elif tipo == "aviso":
                    self.agregar_mensaje("Smith", str(contenido))

        except queue.Empty:
            pass

        if not self.cerrando:
            try:
                if self.root.winfo_exists():
                    self.root.after(100, self.procesar_cola)
            except tk.TclError:
                pass

    def cerrar(self) -> None:
        """Detiene el audio y cierra la interfaz."""
        if self.cerrando:
            return

        self.cerrando = True
        self.detener_grabacion.set()

        try:
            sd.stop()
        except (sd.PortAudioError, RuntimeError):
            pass

        self.root.destroy()


def iniciar_interfaz() -> None:
    """Inicia la interfaz gráfica de Agente Smith."""
    ventana = tk.Tk()
    InterfazSmith(ventana)
    ventana.mainloop()