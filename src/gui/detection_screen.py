"""
DetectionScreen
===============
Pantalla de detección facial y análisis de emociones en tiempo real.

Transmite el video de la webcam, dibuja bounding boxes alrededor de los
rostros detectados, superpone el nombre de la persona (o 'Desconocido')
y la emoción dominante con nivel de confianza. Registra cada evento
de detección en la base de datos.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PIL import Image

import customtkinter as ctk

from src.logic.face_recognizer import FaceRecognizer
from src.logic.emotion_analyzer import EmotionAnalyzer, EMOTION_LABELS_ES

if TYPE_CHECKING:
    from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_UNKNOWN_LABEL = "Desconocido"


class DetectionScreen(ctk.CTkFrame):
    """Pantalla de detección y reconocimiento en vivo.

    Muestra el feed de la webcam con:
    * Bounding boxes de detección facial
    * Etiquetas de identidad (nombre de la BD o 'Desconocido')
    * Overlay de emoción dominante con confianza
    * Las 7 emociones con sus porcentajes en el panel lateral
    * Tiempo de detección en milisegundos

    Parameters
    ----------
    master:
        Widget padre Tk/CTk.
    db_manager:
        Instancia compartida de :class:`~src.database.db_manager.DatabaseManager`.
    """

    _PREVIEW_W = 640
    _PREVIEW_H = 480
    # Ejecutar reconocimiento cada N frames para reducir carga de CPU
    _RECOGNITION_INTERVAL = 10

    def __init__(self, master, db_manager: "DatabaseManager") -> None:
        super().__init__(master)
        self.db_manager = db_manager
        self._recognizer = FaceRecognizer()
        self._analyzer = EmotionAnalyzer()

        self._cap: Optional[cv2.VideoCapture] = None
        self._streaming = False
        self._frame_count = 0

        # Resultados cacheados para superponer entre ciclos de reconocimiento
        self._last_label = ""
        self._last_apellido = ""
        self._last_emotion = ""
        self._last_confidence = 0.0
        self._last_detection_time_ms = 0.0
        self._last_all_scores = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Panel izquierdo: controles y estadísticas en vivo ----
        left = ctk.CTkScrollableFrame(self, width=250)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self._left_panel = left

        ctk.CTkLabel(
            left,
            text="🔍 Detección en Vivo",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=15, pady=(20, 10), sticky="ew")

        # Botones de control
        self._start_btn = ctk.CTkButton(
            left, text="▶ Iniciar Detección", command=self._start_camera
        )
        self._start_btn.grid(row=1, column=0, padx=15, pady=5, sticky="ew")

        self._stop_btn = ctk.CTkButton(
            left,
            text="⏹ Detener",
            command=self._stop_camera,
            state="disabled",
            fg_color="gray",
        )
        self._stop_btn.grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        # --- Sección: Persona detectada ---
        self._add_separator(left, 3)

        ctk.CTkLabel(
            left, text="👤 Persona Detectada",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=4, column=0, padx=15, pady=(10, 2), sticky="w")

        self._person_label = ctk.CTkLabel(
            left,
            text="—",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#3498db",
        )
        self._person_label.grid(row=5, column=0, padx=15, sticky="w")

        # --- Sección: Emoción dominante ---
        ctk.CTkLabel(
            left, text="🎭 Emoción Dominante",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=6, column=0, padx=15, pady=(15, 2), sticky="w")

        self._emotion_label = ctk.CTkLabel(
            left,
            text="—",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f1c40f",
        )
        self._emotion_label.grid(row=7, column=0, padx=15, sticky="w")

        # Barra de confianza
        ctk.CTkLabel(left, text="Confianza:").grid(
            row=8, column=0, padx=15, pady=(5, 0), sticky="w"
        )
        self._confidence_bar = ctk.CTkProgressBar(left, width=200)
        self._confidence_bar.grid(row=9, column=0, padx=15, sticky="ew")
        self._confidence_bar.set(0)

        self._confidence_label = ctk.CTkLabel(left, text="0%")
        self._confidence_label.grid(row=10, column=0, padx=15, sticky="w")

        # Tiempo de detección
        ctk.CTkLabel(left, text="⏱ Tiempo de detección:").grid(
            row=11, column=0, padx=15, pady=(10, 0), sticky="w"
        )
        self._time_label = ctk.CTkLabel(
            left, text="— ms", font=ctk.CTkFont(size=13),
            text_color="#2ecc71",
        )
        self._time_label.grid(row=12, column=0, padx=15, sticky="w")

        # --- Sección: Todas las emociones ---
        self._add_separator(left, 13)

        ctk.CTkLabel(
            left, text="📊 Distribución de Emociones",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=14, column=0, padx=15, pady=(10, 5), sticky="w")

        # Crear labels para las 7 emociones
        self._emotion_bars = {}
        self._emotion_value_labels = {}
        emotions_es = list(EMOTION_LABELS_ES.values())
        base_row = 15
        for i, emotion_es in enumerate(emotions_es):
            row = base_row + i * 2
            ctk.CTkLabel(left, text=emotion_es, font=ctk.CTkFont(size=11)).grid(
                row=row, column=0, padx=15, pady=(2, 0), sticky="w"
            )
            bar_frame = ctk.CTkFrame(left, fg_color="transparent")
            bar_frame.grid(row=row + 1, column=0, padx=15, sticky="ew")

            bar = ctk.CTkProgressBar(bar_frame, width=160, height=12)
            bar.pack(side="left", padx=(0, 5))
            bar.set(0)

            val_label = ctk.CTkLabel(
                bar_frame, text="0%", font=ctk.CTkFont(size=10), width=40
            )
            val_label.pack(side="left")

            self._emotion_bars[emotion_es] = bar
            self._emotion_value_labels[emotion_es] = val_label

        # Estado
        status_row = base_row + len(emotions_es) * 2 + 1
        self._status_label = ctk.CTkLabel(
            left, text="Estado: Listo", wraplength=220
        )
        self._status_label.grid(
            row=status_row, column=0, padx=15, pady=15, sticky="sw"
        )

        # ---- Panel derecho: preview de video ----
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self._preview_label = ctk.CTkLabel(right, text="📷 Sin señal de cámara")
        self._preview_label.pack(expand=True, fill="both", padx=10, pady=10)

    def _add_separator(self, parent, row: int) -> None:
        """Agregar una línea separadora horizontal."""
        sep = ctk.CTkFrame(parent, height=2, fg_color="gray40")
        sep.grid(row=row, column=0, padx=15, pady=10, sticky="ew")

    # ------------------------------------------------------------------
    # Gestión de cámara
    # ------------------------------------------------------------------

    def _start_camera(self) -> None:
        """Iniciar la detección desde la webcam."""
        if self._streaming:
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._set_status("❌ Error: No se pudo abrir la cámara.", error=True)
            return
        self._streaming = True
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._set_status("Transmitiendo… Detectando rostros.")
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def _stop_camera(self) -> None:
        """Detener la captura y liberar la cámara."""
        self._streaming = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._set_status("Cámara detenida.")

    def _stream_loop(self) -> None:
        """Bucle principal de streaming y detección (ejecuta en hilo)."""
        while self._streaming and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            self._frame_count += 1
            annotated = frame.copy()

            # Detectar rostros
            try:
                faces = self._recognizer.detect_faces(frame)
            except Exception:
                faces = []

            if faces:
                face = faces[0]
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]
                face_crop = frame[y : y + h, x : x + w]

                # Ejecutar reconocimiento y emoción cada N frames
                if self._frame_count % self._RECOGNITION_INTERVAL == 0:
                    self._run_analysis(face_crop)

                # Construir overlay
                label = self._last_label or _UNKNOWN_LABEL
                if self._last_apellido:
                    label = f"{label} {self._last_apellido}"
                emotion = self._last_emotion
                confidence_pct = round(self._last_confidence * 100)

                overlay_text = label
                if emotion:
                    overlay_text += f" | {emotion} ({confidence_pct}%)"

                FaceRecognizer.draw_face_boxes(
                    annotated, [face], label=overlay_text, color=(0, 200, 100)
                )

            self._update_preview(annotated)

        self._streaming = False

    # ------------------------------------------------------------------
    # Análisis
    # ------------------------------------------------------------------

    def _run_analysis(self, face_crop: np.ndarray) -> None:
        """Ejecutar reconocimiento facial y análisis de emociones."""
        start_time = time.perf_counter()

        # Identidad
        try:
            match = self._recognizer.identify(face_crop, self.db_manager)
            if match:
                self._last_label = match["name"]
                self._last_apellido = match.get("apellido", "")
                person_id = match["person_id"]
                full_name = f"{match['name']} {match.get('apellido', '')}".strip()
            else:
                self._last_label = _UNKNOWN_LABEL
                self._last_apellido = ""
                person_id = None
                full_name = _UNKNOWN_LABEL
        except Exception as exc:
            logger.error("Error en identificación: %s", exc)
            self._last_label = _UNKNOWN_LABEL
            self._last_apellido = ""
            person_id = None
            full_name = _UNKNOWN_LABEL

        # Emoción
        try:
            emotion_result = self._analyzer.analyze(face_crop)
            if emotion_result:
                self._last_emotion = emotion_result["dominant_emotion_es"]
                self._last_confidence = max(emotion_result["scores"].values())
                self._last_all_scores = emotion_result.get("scores_es", {})
            else:
                self._last_emotion = ""
                self._last_confidence = 0.0
                self._last_all_scores = {}
        except Exception as exc:
            logger.error("Error en análisis de emociones: %s", exc)
            self._last_emotion = ""
            self._last_confidence = 0.0
            self._last_all_scores = {}

        # Tiempo de detección
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._last_detection_time_ms = elapsed_ms

        # Actualizar UI (thread-safe)
        self.after(
            0,
            self._update_stats,
            full_name,
            self._last_emotion,
            self._last_confidence,
            elapsed_ms,
            self._last_all_scores,
        )

        # Persistir log en la base de datos
        try:
            self.db_manager.log_detection(
                person_id=person_id,
                person_name=full_name,
                dominant_emotion=self._last_emotion or None,
                confidence=self._last_confidence if self._last_confidence > 0 else None,
            )
        except Exception as exc:
            logger.error("Error al guardar log: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_stats(
        self,
        name: str,
        emotion: str,
        confidence: float,
        time_ms: float,
        all_scores: dict,
    ) -> None:
        """Actualizar las etiquetas del panel lateral (hilo principal)."""
        # Persona
        self._person_label.configure(text=name or "—")

        # Emoción dominante
        self._emotion_label.configure(text=emotion or "—")

        # Barra y etiqueta de confianza
        self._confidence_bar.set(confidence)
        self._confidence_label.configure(text=f"{round(confidence * 100)}%")

        # Tiempo de detección
        self._time_label.configure(text=f"{time_ms:.0f} ms")

        # Actualizar barras de las 7 emociones
        for emotion_es, bar in self._emotion_bars.items():
            score = all_scores.get(emotion_es, 0.0)
            bar.set(score)
            self._emotion_value_labels[emotion_es].configure(
                text=f"{round(score * 100)}%"
            )

    def _update_preview(self, frame: np.ndarray) -> None:
        """Actualizar la vista previa del video en la UI."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(
            (self._PREVIEW_W, self._PREVIEW_H), Image.LANCZOS
        )
        ctk_img = ctk.CTkImage(
            light_image=img, size=(self._PREVIEW_W, self._PREVIEW_H)
        )
        self._preview_label.configure(image=ctk_img, text="")

    def _set_status(self, message: str, error: bool = False) -> None:
        """Actualizar la etiqueta de estado."""
        color = "#e74c3c" if error else "white"
        self._status_label.configure(text=f"Estado: {message}", text_color=color)
