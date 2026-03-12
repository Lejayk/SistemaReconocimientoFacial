"""
ReportsScreen
=============
Analytics and audit-log screen.

Displays:
* A table of recent detection events (paginated).
* A bar chart of detections per emotion.
* A bar chart of detections per person.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import TYPE_CHECKING, List

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_TABLE_COLUMNS = ("ID", "Persona", "Emoción", "Confianza", "Fecha/Hora")
_PAGE_SIZE = 20


class ReportsScreen(ctk.CTkFrame):
    """Reports and analytics screen.

    Parameters
    ----------
    master:
        Parent Tk/CTk widget.
    db_manager:
        Shared :class:`~src.database.db_manager.DatabaseManager`.
    """

    def __init__(self, master, db_manager: "DatabaseManager") -> None:
        super().__init__(master)
        self.db_manager = db_manager
        self._offset = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---- Header ----
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ctk.CTkLabel(
            header,
            text="Reportes y Estadísticas",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left", padx=15, pady=10)

        ctk.CTkButton(
            header, text="🔄 Actualizar", command=self._refresh, width=120
        ).pack(side="right", padx=15)

        # ---- Tabs ----
        self._tabs = ctk.CTkTabview(self)
        self._tabs.grid(row=1, column=0, sticky="nsew")

        self._tabs.add("📋 Eventos")
        self._tabs.add("😀 Por Emoción")
        self._tabs.add("👤 Por Persona")

        self._build_events_tab(self._tabs.tab("📋 Eventos"))
        self._build_emotion_chart_tab(self._tabs.tab("😀 Por Emoción"))
        self._build_person_chart_tab(self._tabs.tab("👤 Por Persona"))

        self._refresh()

    # ------------------------------------------------------------------
    # Events tab
    # ------------------------------------------------------------------

    def _build_events_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Scrollable frame acting as a simple table
        self._table_frame = ctk.CTkScrollableFrame(parent)
        self._table_frame.grid(row=0, column=0, sticky="nsew")

        # Column headers
        for col, name in enumerate(_TABLE_COLUMNS):
            ctk.CTkLabel(
                self._table_frame,
                text=name,
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=col, padx=8, pady=4, sticky="w")

        # Pagination controls
        nav = ctk.CTkFrame(parent)
        nav.grid(row=1, column=0, sticky="ew", pady=5)

        self._prev_btn = ctk.CTkButton(
            nav, text="◀ Anterior", command=self._prev_page, width=100
        )
        self._prev_btn.pack(side="left", padx=10)

        self._page_label = ctk.CTkLabel(nav, text="Página 1")
        self._page_label.pack(side="left", padx=10)

        self._next_btn = ctk.CTkButton(
            nav, text="Siguiente ▶", command=self._next_page, width=100
        )
        self._next_btn.pack(side="left", padx=10)

    def _populate_table(self, logs: List[dict]) -> None:
        # Remove existing rows (keep header row 0)
        for widget in self._table_frame.grid_slaves():
            if int(widget.grid_info()["row"]) > 0:
                widget.destroy()

        for row_idx, log in enumerate(logs, start=1):
            values = [
                str(log["id"]),
                log["person_name"] or "—",
                log["dominant_emotion"] or "—",
                f"{log['confidence']:.2f}" if log["confidence"] is not None else "—",
                str(log["detected_at"])[:19] if log["detected_at"] else "—",
            ]
            for col_idx, val in enumerate(values):
                ctk.CTkLabel(self._table_frame, text=val).grid(
                    row=row_idx, column=col_idx, padx=8, pady=2, sticky="w"
                )

    # ------------------------------------------------------------------
    # Emotion chart tab
    # ------------------------------------------------------------------

    def _build_emotion_chart_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self._emotion_chart_frame = parent

    def _render_emotion_chart(self) -> None:
        # Clear previous chart
        for w in self._emotion_chart_frame.winfo_children():
            w.destroy()

        summary = self.db_manager.get_emotion_summary()
        if not summary:
            ctk.CTkLabel(
                self._emotion_chart_frame,
                text="Sin datos de emociones aún.",
            ).pack(expand=True)
            return

        emotions = [s["emotion"] for s in summary]
        counts = [s["count"] for s in summary]

        fig = Figure(figsize=(6, 4), dpi=100, facecolor="#2b2b2b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#3b3b3b")
        bars = ax.bar(emotions, counts, color="#1f6aa5")
        ax.set_title("Detecciones por Emoción", color="white")
        ax.set_xlabel("Emoción", color="white")
        ax.set_ylabel("Cantidad", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        canvas = FigureCanvasTkAgg(fig, master=self._emotion_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    # ------------------------------------------------------------------
    # Person chart tab
    # ------------------------------------------------------------------

    def _build_person_chart_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self._person_chart_frame = parent

    def _render_person_chart(self) -> None:
        for w in self._person_chart_frame.winfo_children():
            w.destroy()

        logs = self.db_manager.get_detection_logs(limit=1000)
        if not logs:
            ctk.CTkLabel(
                self._person_chart_frame,
                text="Sin datos de personas aún.",
            ).pack(expand=True)
            return

        counts: dict = {}
        for log in logs:
            name = log["person_name"] or "Desconocido"
            counts[name] = counts.get(name, 0) + 1

        names = list(counts.keys())
        values = [counts[n] for n in names]

        fig = Figure(figsize=(6, 4), dpi=100, facecolor="#2b2b2b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#3b3b3b")
        ax.barh(names, values, color="#1f6aa5")
        ax.set_title("Detecciones por Persona", color="white")
        ax.set_xlabel("Cantidad", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._person_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    def _prev_page(self) -> None:
        if self._offset >= _PAGE_SIZE:
            self._offset -= _PAGE_SIZE
            self._refresh_table()

    def _next_page(self) -> None:
        self._offset += _PAGE_SIZE
        self._refresh_table()

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        logs = self.db_manager.get_detection_logs(
            limit=_PAGE_SIZE, offset=self._offset
        )
        self._populate_table(logs)
        page = self._offset // _PAGE_SIZE + 1
        self._page_label.configure(text=f"Página {page}")
        self._prev_btn.configure(state="normal" if self._offset > 0 else "disabled")
        self._next_btn.configure(state="normal" if len(logs) == _PAGE_SIZE else "disabled")

    def _refresh(self) -> None:
        self._offset = 0
        self._refresh_table()
        self._render_emotion_chart()
        self._render_person_chart()
