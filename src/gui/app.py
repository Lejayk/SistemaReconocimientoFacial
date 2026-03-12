"""
App – Main Application Window
==============================
Root CustomTkinter window that hosts the three screens:
    - RegistrationScreen
    - DetectionScreen
    - ReportsScreen

Navigation is handled via a sidebar with icon buttons.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.gui.registration_screen import RegistrationScreen
from src.gui.detection_screen import DetectionScreen
from src.gui.reports_screen import ReportsScreen

if TYPE_CHECKING:
    from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Appearance defaults
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_APP_TITLE = "Sistema de Reconocimiento Facial"
_WINDOW_SIZE = "1100x700"


class App(ctk.CTk):
    """Root window that orchestrates screen navigation.

    Parameters
    ----------
    db_manager:
        Shared :class:`~src.database.db_manager.DatabaseManager` instance
        passed down to every screen.
    """

    def __init__(self, db_manager: "DatabaseManager") -> None:
        super().__init__()

        self.db_manager = db_manager
        self.title(_APP_TITLE)
        self.geometry(_WINDOW_SIZE)
        self.minsize(900, 600)

        self._build_layout()
        self._build_sidebar()
        self._build_screens()

        # Show the registration screen by default
        self.show_screen("registration")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _build_sidebar(self) -> None:
        self._sidebar = ctk.CTkFrame(self, width=180, corner_radius=0)
        self._sidebar.grid(row=0, column=0, sticky="nsew")
        self._sidebar.grid_rowconfigure(10, weight=1)

        # App logo / title
        logo_label = ctk.CTkLabel(
            self._sidebar,
            text="🎭 FaceID",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Navigation buttons
        nav_items = [
            ("📝 Registro", "registration"),
            ("🔍 Detección", "detection"),
            ("📊 Reportes", "reports"),
        ]
        for i, (label, screen) in enumerate(nav_items, start=1):
            btn = ctk.CTkButton(
                self._sidebar,
                text=label,
                command=lambda s=screen: self.show_screen(s),
                anchor="w",
            )
            btn.grid(row=i, column=0, padx=20, pady=5, sticky="ew")

        # Appearance mode selector
        appearance_label = ctk.CTkLabel(self._sidebar, text="Tema:")
        appearance_label.grid(row=11, column=0, padx=20, pady=(10, 0))
        appearance_menu = ctk.CTkOptionMenu(
            self._sidebar,
            values=["dark", "light", "system"],
            command=ctk.set_appearance_mode,
        )
        appearance_menu.grid(row=12, column=0, padx=20, pady=(5, 20))

    def _build_screens(self) -> None:
        shared_kwargs = {"master": self, "db_manager": self.db_manager}
        self._screens = {
            "registration": RegistrationScreen(**shared_kwargs),
            "detection": DetectionScreen(**shared_kwargs),
            "reports": ReportsScreen(**shared_kwargs),
        }
        for screen in self._screens.values():
            screen.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def show_screen(self, name: str) -> None:
        """Raise the screen with the given *name* to the front.

        Parameters
        ----------
        name:
            One of ``"registration"``, ``"detection"``, ``"reports"``.
        """
        screen = self._screens.get(name)
        if screen is None:
            logger.warning("Unknown screen: '%s'", name)
            return
        screen.tkraise()
        logger.debug("Switched to screen: '%s'", name)
