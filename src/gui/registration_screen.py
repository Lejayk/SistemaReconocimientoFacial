"""
RegistrationScreen
==================
Screen for capturing a face from the webcam (or loading an image) and
registering a new person in the database.
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

if TYPE_CHECKING:
    from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class RegistrationScreen(ctk.CTkFrame):
    """Face registration screen.

    Allows the user to:
    1. Enter a name for the person to register.
    2. Capture a frame from the live webcam feed.
    3. Confirm registration (stores embedding in the DB).

    Parameters
    ----------
    master:
        Parent Tk/CTk widget.
    db_manager:
        Shared :class:`~src.database.db_manager.DatabaseManager`.
    """

    _PREVIEW_W = 480
    _PREVIEW_H = 360

    def __init__(self, master, db_manager: "DatabaseManager") -> None:
        super().__init__(master)
        self.db_manager = db_manager
        self._recognizer = FaceRecognizer()

        self._cap: Optional[cv2.VideoCapture] = None
        self._current_frame: Optional[np.ndarray] = None
        self._streaming = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Left panel: controls ----
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            left, text="Registro de Persona", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=15, pady=(20, 10))

        ctk.CTkLabel(left, text="Nombre:").grid(
            row=1, column=0, padx=15, pady=5, sticky="w"
        )
        self._name_entry = ctk.CTkEntry(left, placeholder_text="Nombre completo")
        self._name_entry.grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        self._start_btn = ctk.CTkButton(
            left, text="▶ Iniciar Cámara", command=self._start_camera
        )
        self._start_btn.grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        self._capture_btn = ctk.CTkButton(
            left,
            text="📸 Capturar",
            command=self._capture_frame,
            state="disabled",
        )
        self._capture_btn.grid(row=4, column=0, padx=15, pady=5, sticky="ew")

        self._register_btn = ctk.CTkButton(
            left,
            text="✅ Registrar",
            command=self._register_person,
            state="disabled",
            fg_color="green",
            hover_color="darkgreen",
        )
        self._register_btn.grid(row=5, column=0, padx=15, pady=5, sticky="ew")

        self._stop_btn = ctk.CTkButton(
            left,
            text="⏹ Detener Cámara",
            command=self._stop_camera,
            state="disabled",
            fg_color="gray",
        )
        self._stop_btn.grid(row=6, column=0, padx=15, pady=5, sticky="ew")

        self._status_label = ctk.CTkLabel(
            left, text="Estado: Listo", wraplength=200
        )
        self._status_label.grid(row=7, column=0, padx=15, pady=10, sticky="w")

        # ---- Right panel: live preview ----
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
        self._capture_btn.configure(state="normal")
        self._stop_btn.configure(state="normal")
        self._start_btn.configure(state="disabled")
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def _stop_camera(self) -> None:
        self._streaming = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._start_btn.configure(state="normal")
        self._capture_btn.configure(state="disabled")
        self._stop_btn.configure(state="disabled")
        self._set_status("Cámara detenida.")

    def _stream_loop(self) -> None:
        while self._streaming and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            self._current_frame = frame.copy()
            self._update_preview(frame)
        self._streaming = False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _capture_frame(self) -> None:
        if self._current_frame is None:
            self._set_status("No hay frame disponible.", error=True)
            return
        self._captured_frame = self._current_frame.copy()
        self._register_btn.configure(state="normal")
        self._set_status("Frame capturado. Presiona 'Registrar' para guardar.")

    def _register_person(self) -> None:
        name = self._name_entry.get().strip()
        if not name:
            self._set_status("Por favor ingresa un nombre.", error=True)
            return
        if not hasattr(self, "_captured_frame") or self._captured_frame is None:
            self._set_status("Captura un frame primero.", error=True)
            return

        success = self._recognizer.register_face(
            name=name,
            face_img=self._captured_frame,
            db_manager=self.db_manager,
        )
        if success:
            self._set_status(f"✅ '{name}' registrado exitosamente.")
            self._register_btn.configure(state="disabled")
            self._name_entry.delete(0, "end")
        else:
            self._set_status("Error al registrar. ¿Se detectó un rostro?", error=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_preview(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(
            (self._PREVIEW_W, self._PREVIEW_H), Image.LANCZOS
        )
        ctk_img = ctk.CTkImage(light_image=img, size=(self._PREVIEW_W, self._PREVIEW_H))
        self._preview_label.configure(image=ctk_img, text="")

    def _set_status(self, message: str, error: bool = False) -> None:
        color = "red" if error else "white"
        self._status_label.configure(text=f"Estado: {message}", text_color=color)
