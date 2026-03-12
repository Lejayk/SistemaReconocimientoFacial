"""
DetectionScreen
===============
Real-time face detection, recognition and emotion analysis screen.

Streams the webcam feed, draws bounding boxes around detected faces,
overlays the person's name (or 'Desconocido') and the dominant emotion,
and logs each detection event to the database.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PIL import Image

import customtkinter as ctk

from src.logic.face_recognizer import FaceRecognizer
from src.logic.emotion_analyzer import EmotionAnalyzer

if TYPE_CHECKING:
    from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_UNKNOWN_LABEL = "Desconocido"


class DetectionScreen(ctk.CTkFrame):
    """Live detection and recognition screen.

    Shows the webcam feed with real-time:
    * Face detection bounding boxes
    * Identity labels (name from DB or 'Desconocido')
    * Dominant emotion overlay
    * Confidence score

    Parameters
    ----------
    master:
        Parent Tk/CTk widget.
    db_manager:
        Shared :class:`~src.database.db_manager.DatabaseManager`.
    """

    _PREVIEW_W = 640
    _PREVIEW_H = 480
    # Run recognition every N frames to reduce CPU load
    _RECOGNITION_INTERVAL = 10

    def __init__(self, master, db_manager: "DatabaseManager") -> None:
        super().__init__(master)
        self.db_manager = db_manager
        self._recognizer = FaceRecognizer()
        self._analyzer = EmotionAnalyzer()

        self._cap: Optional[cv2.VideoCapture] = None
        self._streaming = False
        self._frame_count = 0

        # Cached results to overlay between recognition cycles
        self._last_label = ""
        self._last_emotion = ""

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Left panel: controls & live stats ----
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            left, text="Detección en Vivo", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, padx=15, pady=(20, 10))

        self._start_btn = ctk.CTkButton(
            left, text="▶ Iniciar", command=self._start_camera
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

        ctk.CTkLabel(left, text="Persona detectada:").grid(
            row=3, column=0, padx=15, pady=(20, 2), sticky="w"
        )
        self._person_label = ctk.CTkLabel(
            left,
            text="—",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="cyan",
        )
        self._person_label.grid(row=4, column=0, padx=15, sticky="w")

        ctk.CTkLabel(left, text="Emoción dominante:").grid(
            row=5, column=0, padx=15, pady=(15, 2), sticky="w"
        )
        self._emotion_label = ctk.CTkLabel(
            left,
            text="—",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="yellow",
        )
        self._emotion_label.grid(row=6, column=0, padx=15, sticky="w")

        self._status_label = ctk.CTkLabel(left, text="Estado: Listo", wraplength=180)
        self._status_label.grid(row=11, column=0, padx=15, pady=20, sticky="sw")

        # ---- Right panel: video preview ----
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self._preview_label = ctk.CTkLabel(right, text="Sin señal de cámara")
        self._preview_label.pack(expand=True, fill="both", padx=10, pady=10)

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def _start_camera(self) -> None:
        if self._streaming:
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._set_status("Error: No se pudo abrir la cámara.", error=True)
            return
        self._streaming = True
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._set_status("Transmitiendo…")
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def _stop_camera(self) -> None:
        self._streaming = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._set_status("Cámara detenida.")

    def _stream_loop(self) -> None:
        while self._streaming and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            self._frame_count += 1
            annotated = frame.copy()

            # Detect faces
            faces = self._recognizer.detect_faces(frame)

            if faces:
                face = faces[0]
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]
                face_crop = frame[y : y + h, x : x + w]

                # Run recognition / emotion every N frames
                if self._frame_count % self._RECOGNITION_INTERVAL == 0:
                    self._run_analysis(face_crop)

                label = self._last_label or _UNKNOWN_LABEL
                emotion = self._last_emotion

                overlay = f"{label} | {emotion}" if emotion else label
                FaceRecognizer.draw_face_boxes(
                    annotated, [face], label=overlay, color=(0, 200, 100)
                )

            self._update_preview(annotated)

        self._streaming = False

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _run_analysis(self, face_crop: np.ndarray) -> None:
        """Run recognition and emotion analysis on *face_crop*."""
        # Identity
        match = self._recognizer.identify(face_crop, self.db_manager)
        if match:
            self._last_label = match["name"]
            person_id = match["person_id"]
        else:
            self._last_label = _UNKNOWN_LABEL
            person_id = None

        # Emotion
        emotion_result = self._analyzer.analyze(face_crop)
        if emotion_result:
            self._last_emotion = emotion_result["dominant_emotion_es"]
            confidence = max(emotion_result["scores"].values())
        else:
            self._last_emotion = ""
            confidence = None

        # Update sidebar labels (thread-safe via after)
        self.after(0, self._update_stats, self._last_label, self._last_emotion)

        # Persist log
        self.db_manager.log_detection(
            person_id=person_id,
            person_name=self._last_label,
            dominant_emotion=self._last_emotion or None,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_stats(self, name: str, emotion: str) -> None:
        self._person_label.configure(text=name or "—")
        self._emotion_label.configure(text=emotion or "—")

    def _update_preview(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(
            (self._PREVIEW_W, self._PREVIEW_H), Image.LANCZOS
        )
        ctk_img = ctk.CTkImage(
            light_image=img, size=(self._PREVIEW_W, self._PREVIEW_H)
        )
        self._preview_label.configure(image=ctk_img, text="")

    def _set_status(self, message: str, error: bool = False) -> None:
        color = "red" if error else "white"
        self._status_label.configure(text=f"Estado: {message}", text_color=color)
