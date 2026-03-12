"""
RegistrationScreen
==================
Pantalla para capturar rostros desde la webcam y registrar
personas nuevas en la base de datos.

Incluye campos para nombre, apellido y email, captura múltiple
con indicador de calidad, y validación de duplicados.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np
from PIL import Image

import customtkinter as ctk

from src.logic.face_recognizer import FaceRecognizer

if TYPE_CHECKING:
    from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Número máximo de capturas para registro
_MAX_CAPTURES = 5


class RegistrationScreen(ctk.CTkFrame):
    """Pantalla de registro facial.

    Permite al usuario:
    1. Ingresar nombre, apellido y email.
    2. Capturar múltiples frames desde la webcam en vivo.
    3. Ver indicador de calidad por cada captura.
    4. Confirmar el registro (almacena embedding en la BD).

    Parameters
    ----------
    master:
        Widget padre Tk/CTk.
    db_manager:
        Instancia compartida de :class:`~src.database.db_manager.DatabaseManager`.
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

        # Lista de rostros capturados para registro múltiple
        self._captured_faces: List[np.ndarray] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Panel izquierdo: controles ----
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.grid_rowconfigure(20, weight=1)

        ctk.CTkLabel(
            left,
            text="📝 Registro de Persona",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=15, pady=(20, 15))

        # Campo: Nombre
        ctk.CTkLabel(left, text="Nombre: *").grid(
            row=1, column=0, padx=15, pady=(5, 0), sticky="w"
        )
        self._name_entry = ctk.CTkEntry(left, placeholder_text="Ej: Juan", width=220)
        self._name_entry.grid(row=2, column=0, padx=15, pady=(0, 5), sticky="ew")

        # Campo: Apellido
        ctk.CTkLabel(left, text="Apellido: *").grid(
            row=3, column=0, padx=15, pady=(5, 0), sticky="w"
        )
        self._apellido_entry = ctk.CTkEntry(
            left, placeholder_text="Ej: Pérez", width=220
        )
        self._apellido_entry.grid(row=4, column=0, padx=15, pady=(0, 5), sticky="ew")

        # Campo: Email
        ctk.CTkLabel(left, text="Email:").grid(
            row=5, column=0, padx=15, pady=(5, 0), sticky="w"
        )
        self._email_entry = ctk.CTkEntry(
            left, placeholder_text="Ej: juan@email.com", width=220
        )
        self._email_entry.grid(row=6, column=0, padx=15, pady=(0, 10), sticky="ew")

        # Botones de acción
        self._start_btn = ctk.CTkButton(
            left, text="▶ Iniciar Cámara", command=self._start_camera
        )
        self._start_btn.grid(row=7, column=0, padx=15, pady=5, sticky="ew")

        self._capture_btn = ctk.CTkButton(
            left,
            text="📸 Capturar (0/5)",
            command=self._capture_frame,
            state="disabled",
        )
        self._capture_btn.grid(row=8, column=0, padx=15, pady=5, sticky="ew")

        self._register_btn = ctk.CTkButton(
            left,
            text="✅ Registrar Persona",
            command=self._register_person,
            state="disabled",
            fg_color="#2d8f2d",
            hover_color="#1e6b1e",
        )
        self._register_btn.grid(row=9, column=0, padx=15, pady=5, sticky="ew")

        self._stop_btn = ctk.CTkButton(
            left,
            text="⏹ Detener Cámara",
            command=self._stop_camera,
            state="disabled",
            fg_color="gray",
        )
        self._stop_btn.grid(row=10, column=0, padx=15, pady=5, sticky="ew")

        self._reset_btn = ctk.CTkButton(
            left,
            text="🔄 Limpiar Capturas",
            command=self._reset_captures,
            fg_color="#8B4513",
            hover_color="#654321",
        )
        self._reset_btn.grid(row=11, column=0, padx=15, pady=5, sticky="ew")

        # Indicador de calidad
        ctk.CTkLabel(
            left, text="Calidad del rostro:", font=ctk.CTkFont(size=12)
        ).grid(row=12, column=0, padx=15, pady=(15, 2), sticky="w")

        self._quality_bar = ctk.CTkProgressBar(left, width=200)
        self._quality_bar.grid(row=13, column=0, padx=15, pady=(0, 2), sticky="ew")
        self._quality_bar.set(0)

        self._quality_label = ctk.CTkLabel(
            left, text="⚪ Sin captura", font=ctk.CTkFont(size=12)
        )
        self._quality_label.grid(row=14, column=0, padx=15, pady=(0, 5), sticky="w")

        # ---- Sección: Personas registradas con opción de eliminar ----
        sep = ctk.CTkFrame(left, height=2, fg_color="gray40")
        sep.grid(row=15, column=0, padx=15, pady=10, sticky="ew")

        ctk.CTkLabel(
            left, text="👥 Personas Registradas",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=16, column=0, padx=15, pady=(0, 5), sticky="w")

        self._persons_list_frame = ctk.CTkScrollableFrame(
            left, width=200, height=120
        )
        self._persons_list_frame.grid(row=17, column=0, padx=15, pady=(0, 5), sticky="ew")

        ctk.CTkButton(
            left, text="🔄 Actualizar Lista",
            command=self._refresh_persons_list,
            width=120, height=28, font=ctk.CTkFont(size=12),
        ).grid(row=18, column=0, padx=15, pady=(0, 5), sticky="ew")

        # Mensaje de estado
        self._status_label = ctk.CTkLabel(
            left, text="Estado: Listo", wraplength=220
        )
        self._status_label.grid(row=21, column=0, padx=15, pady=10, sticky="sw")
        
        # Populate the list on initialization
        self.after(500, self._refresh_persons_list)

        # ---- Panel derecho: preview en vivo ----
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self._preview_label = ctk.CTkLabel(right, text="📷 Sin señal de cámara")
        self._preview_label.pack(expand=True, fill="both", padx=10, pady=10)

    # ------------------------------------------------------------------
    # Gestión de cámara
    # ------------------------------------------------------------------

    def _start_camera(self) -> None:
        """Iniciar la captura de video desde la webcam."""
        if self._streaming:
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._set_status("❌ Error: No se pudo abrir la cámara.", error=True)
            return
        self._streaming = True
        self._capture_btn.configure(state="normal")
        self._stop_btn.configure(state="normal")
        self._start_btn.configure(state="disabled")
        self._set_status("Cámara iniciada. Posiciona tu rostro y captura.")
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def _stop_camera(self) -> None:
        """Detener la captura de video y liberar la cámara."""
        self._streaming = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._start_btn.configure(state="normal")
        self._capture_btn.configure(state="disabled")
        self._stop_btn.configure(state="disabled")
        self._set_status("Cámara detenida.")

    def _stream_loop(self) -> None:
        """Bucle principal de streaming de la cámara (ejecuta en hilo)."""
        while self._streaming and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            # Detectar rostros para dibujar rectángulos en preview
            self._current_frame = frame.copy()
            annotated = frame.copy()

            try:
                # Usar detección rápida (Haar Cascade) para el preview en vivo
                faces = self._recognizer.detect_faces_fast(frame)
                if faces:
                    face = faces[0]
                    x, y, w, h = face["x"], face["y"], face["w"], face["h"]

                    # Evaluar calidad en tiempo real
                    face_crop = frame[y : y + h, x : x + w]
                    if face_crop.size > 0:
                        quality = self._recognizer.assess_face_quality(face_crop)
                        self.after(0, self._update_quality_indicator, quality)

                    FaceRecognizer.draw_face_boxes(
                        annotated, [face], label="Rostro detectado", color=(0, 200, 100)
                    )
            except Exception:
                pass

            self._update_preview(annotated)

        self._streaming = False

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------

    def _capture_frame(self) -> None:
        """Capturar el frame actual y añadirlo a la lista de capturas."""
        if self._current_frame is None:
            self._set_status("❌ No hay frame disponible.", error=True)
            return

        # Detectar y recortar el rostro (usar detección rápida para captura)
        faces = self._recognizer.detect_faces_fast(self._current_frame)
        if not faces:
            self._set_status("❌ No se detectó ningún rostro. Intenta de nuevo.", error=True)
            return

        face = faces[0]
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        face_crop = self._current_frame[y : y + h, x : x + w]

        if face_crop.size == 0:
            self._set_status("❌ Rostro fuera de cuadro.", error=True)
            return

        # Evaluar calidad
        quality = self._recognizer.assess_face_quality(face_crop)
        if not quality["is_valid"]:
            issues = []
            if not quality["size_ok"]:
                issues.append("rostro muy pequeño")
            if not quality["sharpness_ok"]:
                issues.append("imagen borrosa")
            self._set_status(
                f"⚠️ Calidad insuficiente: {', '.join(issues)}. "
                "Acércate más a la cámara.",
                error=True,
            )
            return

        self._captured_faces.append(face_crop.copy())
        count = len(self._captured_faces)

        self._capture_btn.configure(text=f"📸 Capturar ({count}/{_MAX_CAPTURES})")
        self._set_status(f"✅ Captura {count}/{_MAX_CAPTURES} guardada (calidad: {quality['score']}%)")

        if count >= _MAX_CAPTURES:
            self._capture_btn.configure(state="disabled")
            self._register_btn.configure(state="normal")
            self._set_status(
                f"✅ {_MAX_CAPTURES} capturas completas. "
                "Presiona 'Registrar Persona' para guardar."
            )
        elif count >= 1:
            self._register_btn.configure(state="normal")

    def _register_person(self) -> None:
        """Registrar la persona con los datos ingresados y las capturas."""
        name = self._name_entry.get().strip()
        apellido = self._apellido_entry.get().strip()
        email = self._email_entry.get().strip()

        # Validaciones
        if not name:
            self._set_status("❌ Por favor ingresa el nombre.", error=True)
            return
        if not apellido:
            self._set_status("❌ Por favor ingresa el apellido.", error=True)
            return
        if not self._captured_faces:
            self._set_status("❌ Captura al menos un frame primero.", error=True)
            return

        # Verificar duplicados por email
        if email and self.db_manager.person_exists(email):
            self._set_status(
                f"❌ Ya existe una persona registrada con el email '{email}'.",
                error=True,
            )
            return

        # Registrar con la mejor captura
        self._set_status("⏳ Registrando persona... (extrayendo embedding)")
        self._register_btn.configure(state="disabled")

        def _do_register():
            success = self._recognizer.register_multiple_faces(
                name=name,
                apellido=apellido,
                email=email or None,
                face_imgs=self._captured_faces,
                db_manager=self.db_manager,
            )
            self.after(0, self._on_registration_complete, success, name, apellido)

        threading.Thread(target=_do_register, daemon=True).start()

    def _on_registration_complete(
        self, success: bool, name: str, apellido: str
    ) -> None:
        """Callback tras completar el registro (ejecuta en hilo principal)."""
        if success:
            full_name = f"{name} {apellido}"
            self._set_status(f"✅ '{full_name}' registrado exitosamente.")
            self._recognizer.invalidate_cache()  # Invalidar caché de embeddings
            self._clear_form()
            self._refresh_persons_list()  # Actualizar lista de personas
        else:
            self._set_status(
                "❌ Error al registrar. ¿Se detectó un rostro válido?",
                error=True,
            )
            self._register_btn.configure(state="normal")

    def _reset_captures(self) -> None:
        """Limpiar todas las capturas realizadas."""
        self._captured_faces.clear()
        self._capture_btn.configure(text="📸 Capturar (0/5)", state="normal" if self._streaming else "disabled")
        self._register_btn.configure(state="disabled")
        self._quality_bar.set(0)
        self._quality_label.configure(text="⚪ Sin captura")
        self._set_status("Capturas limpiadas. Listo para nuevas capturas.")

    def _clear_form(self) -> None:
        """Limpiar todos los campos y capturas tras registro exitoso."""
        self._name_entry.delete(0, "end")
        self._apellido_entry.delete(0, "end")
        self._email_entry.delete(0, "end")
        self._reset_captures()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_quality_indicator(self, quality: dict) -> None:
        """Actualizar el indicador visual de calidad del rostro."""
        score = quality["score"]
        self._quality_bar.set(score / 100.0)

        if score >= 70:
            indicator = "🟢 Excelente"
            color = "#2ecc71"
        elif score >= 40:
            indicator = "🟡 Aceptable"
            color = "#f1c40f"
        else:
            indicator = "🔴 Baja calidad"
            color = "#e74c3c"

        self._quality_label.configure(
            text=f"{indicator} ({score}%)", text_color=color
        )

    def _update_preview(self, frame: np.ndarray) -> None:
        """Actualizar la vista previa de la cámara en la UI."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(
            (self._PREVIEW_W, self._PREVIEW_H), Image.LANCZOS
        )
        ctk_img = ctk.CTkImage(
            light_image=img, size=(self._PREVIEW_W, self._PREVIEW_H)
        )
        self._preview_label.configure(image=ctk_img, text="")

    def _set_status(self, message: str, error: bool = False) -> None:
        """Actualizar la etiqueta de estado con un mensaje."""
        color = "#e74c3c" if error else "white"
        self._status_label.configure(text=f"Estado: {message}", text_color=color)

    # ------------------------------------------------------------------
    # Gestión de personas registradas
    # ------------------------------------------------------------------

    def _refresh_persons_list(self) -> None:
        """Recargar la lista de personas registradas con botones de eliminar."""
        # Limpiar lista actual
        for widget in self._persons_list_frame.winfo_children():
            widget.destroy()

        people = self.db_manager.get_all_people()
        if not people:
            ctk.CTkLabel(
                self._persons_list_frame,
                text="No hay personas registradas.",
                font=ctk.CTkFont(size=11),
                text_color="gray60",
            ).pack(padx=5, pady=5)
            return

        for person in people:
            full_name = f"{person['name']} {person.get('apellido', '')}".strip()
            row_frame = ctk.CTkFrame(
                self._persons_list_frame, fg_color="transparent"
            )
            row_frame.pack(fill="x", padx=2, pady=2)

            ctk.CTkLabel(
                row_frame, text=full_name,
                font=ctk.CTkFont(size=11), anchor="w",
            ).pack(side="left", fill="x", expand=True, padx=(5, 2))

            ctk.CTkButton(
                row_frame,
                text="🗑",
                width=30, height=24,
                fg_color="#c0392b",
                hover_color="#962d22",
                font=ctk.CTkFont(size=12),
                command=lambda pid=person["id"], pn=full_name: self._delete_person(pid, pn),
            ).pack(side="right", padx=2)

    def _delete_person(self, person_id: int, person_name: str) -> None:
        """Eliminar una persona registrada tras confirmación."""
        # Crear un diálogo de confirmación
        dialog = ctk.CTkToplevel(self)
        dialog.title("Confirmar eliminación")
        dialog.geometry("380x150")
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text=f"¿Eliminar a '{person_name}'?\n\nEsta acción no se puede deshacer.",
            wraplength=340,
            font=ctk.CTkFont(size=13),
        ).pack(expand=True, padx=20, pady=15)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=(0, 15))

        def _confirm():
            dialog.destroy()
            success = self.db_manager.delete_person(person_id)
            if success:
                self._recognizer.invalidate_cache()
                self._set_status(f"✅ '{person_name}' eliminado correctamente.")
                self._refresh_persons_list()
            else:
                self._set_status(f"❌ No se pudo eliminar a '{person_name}'.", error=True)

        ctk.CTkButton(
            btn_frame, text="❌ Eliminar", command=_confirm,
            fg_color="#c0392b", hover_color="#962d22", width=100,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_frame, text="Cancelar", command=dialog.destroy,
            fg_color="gray", width=100,
        ).pack(side="left", padx=10)
